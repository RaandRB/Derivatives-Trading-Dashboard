"""FX option pricing with SABR vol model.

MARKET PRACTICE:
- FX options are quoted in delta-space: ATM DNS, 25Δ RR, 25Δ BF, 10Δ RR, 10Δ BF
- ATM = delta-neutral straddle (NOT spot or forward ATM)
- Premium currency conventions vary by pair (we use domestic/foreign standard)
- Pricing uses Garman-Kohlhagen (BS extended to two interest rates)
- SABR is calibrated per expiry from the 5 market quotes

CONVERSION: delta-space → strike-space:
  25Δ call vol = ATM + BF25 + RR25/2
  25Δ put vol  = ATM + BF25 - RR25/2
  Then invert the GK delta formula to get strikes from deltas.
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import brentq
from pysabr import Hagan2002LognormalSABR

from dashboard.data.fx import expiry_to_years


@dataclass
class FXOptionResult:
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho_dom: float
    rho_for: float
    vanna: float
    volga: float
    vol: float
    strike: float


def delta_space_to_strikes(
    spot: float, atm: float, rr25: float, bf25: float,
    rd: float, rf: float, T: float,
) -> dict:
    """Convert delta-space vol quotes to strikes and vols.

    Returns dict with keys: atm_strike, c25_strike, p25_strike, c25_vol, p25_vol, atm_vol
    """
    c25_vol = atm + bf25 + rr25 / 2
    p25_vol = atm + bf25 - rr25 / 2

    # ATM DNS strike: strike where straddle is delta-neutral
    # For GK: call_delta + put_delta = 0 → K = F * exp(0.5 * σ² * T)
    F = spot * np.exp((rd - rf) * T)
    atm_strike = F * np.exp(0.5 * atm ** 2 * T)

    # 25-delta strikes via GK delta inversion
    c25_strike = _strike_from_delta(0.25, spot, T, rd, rf, c25_vol, is_call=True)
    p25_strike = _strike_from_delta(-0.25, spot, T, rd, rf, p25_vol, is_call=False)

    return {
        "atm_strike": atm_strike, "atm_vol": atm,
        "c25_strike": c25_strike, "c25_vol": c25_vol,
        "p25_strike": p25_strike, "p25_vol": p25_vol,
    }


def calibrate_sabr_fx(
    spot: float, vol_quotes: dict, T: float, rd: float = 0.04, rf: float = 0.03,
) -> dict:
    """Calibrate SABR for a single FX expiry.

    Args:
        spot: FX spot rate
        vol_quotes: dict with atm, rr25, bf25
        T: time to expiry in years
        rd: domestic risk-free rate
        rf: foreign risk-free rate

    Returns:
        dict with alpha, beta, rho, nu, forward
    """
    F = spot * np.exp((rd - rf) * T)
    atm = vol_quotes["atm"]
    rr25 = vol_quotes.get("rr25", 0.0)
    bf25 = vol_quotes.get("bf25", 0.0)

    strikes_data = delta_space_to_strikes(spot, atm, rr25, bf25, rd, rf, T)

    # Calibrate SABR with beta=1 (lognormal backbone, standard for FX)
    beta = 1.0
    strikes = [strikes_data["p25_strike"], strikes_data["atm_strike"], strikes_data["c25_strike"]]
    vols = [strikes_data["p25_vol"], strikes_data["atm_vol"], strikes_data["c25_vol"]]

    # Use pysabr calibration
    try:
        sabr = Hagan2002LognormalSABR(F, 0.0, T)
        # Start with alpha ≈ atm_vol, then calibrate rho/nu from smile
        alpha = atm
        # Simple grid search for rho and nu
        best_err = float("inf")
        best_params = {"alpha": alpha, "rho": -0.1, "nu": 0.3}

        for rho in np.linspace(-0.5, 0.5, 11):
            for nu in np.linspace(0.1, 1.0, 10):
                # Calibrate alpha to match ATM
                try:
                    a = sabr.fit(F, [strikes_data["atm_strike"]], [atm], beta, rho, nu)[0]
                except Exception:
                    a = alpha
                # Check smile fit
                err = 0
                for k, v in zip(strikes, vols):
                    try:
                        model_v = sabr.lognormalvol(k, T, a, beta, rho, nu)
                        err += (model_v - v) ** 2
                    except Exception:
                        err += 1.0
                if err < best_err:
                    best_err = err
                    best_params = {"alpha": a, "beta": beta, "rho": rho, "nu": nu}
    except Exception:
        best_params = {"alpha": atm, "beta": 1.0, "rho": -0.1, "nu": 0.3}

    best_params["forward"] = F
    return best_params


def price_fx_option(
    spot: float, strike: float, T: float, is_call: bool,
    vol: float, rd: float = 0.04, rf: float = 0.03, notional: float = 1e6,
) -> FXOptionResult:
    """Price an FX option using Garman-Kohlhagen + compute Greeks.

    All Greeks computed via bump-and-reprice for consistency with the approach.
    """
    def gk_price(s, k, t, v, r_d, r_f, call):
        if t <= 0 or v <= 0:
            payoff = (s - k) if call else (k - s)
            return max(0, payoff) * np.exp(-r_d * t) if t > 0 else max(0, payoff)
        d1 = (np.log(s / k) + (r_d - r_f + 0.5 * v ** 2) * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        if call:
            return s * np.exp(-r_f * t) * norm.cdf(d1) - k * np.exp(-r_d * t) * norm.cdf(d2)
        else:
            return k * np.exp(-r_d * t) * norm.cdf(-d2) - s * np.exp(-r_f * t) * norm.cdf(-d1)

    base = gk_price(spot, strike, T, vol, rd, rf, is_call)
    ds = spot * 0.005
    dv = 0.01
    dt = 1 / 365
    dr = 0.0001

    pv_up = gk_price(spot + ds, strike, T, vol, rd, rf, is_call)
    pv_dn = gk_price(spot - ds, strike, T, vol, rd, rf, is_call)
    delta = (pv_up - pv_dn) / (2 * ds)
    gamma = (pv_up - 2 * base + pv_dn) / ds ** 2

    vega = (gk_price(spot, strike, T, vol + dv, rd, rf, is_call) - gk_price(spot, strike, T, vol - dv, rd, rf, is_call)) / 2
    theta = (gk_price(spot, strike, T - dt, vol, rd, rf, is_call) - base) if T > dt else -base
    rho_dom = (gk_price(spot, strike, T, vol, rd + dr, rf, is_call) - base) / dr * 0.01
    rho_for = (gk_price(spot, strike, T, vol, rd, rf + dr, is_call) - base) / dr * 0.01

    # Vanna & Volga
    pv_s_up_v_up = gk_price(spot + ds, strike, T, vol + dv, rd, rf, is_call)
    pv_s_dn_v_up = gk_price(spot - ds, strike, T, vol + dv, rd, rf, is_call)
    delta_v_up = (pv_s_up_v_up - pv_s_dn_v_up) / (2 * ds)
    vanna = (delta_v_up - delta) / dv

    pv_v_up = gk_price(spot, strike, T, vol + dv, rd, rf, is_call)
    pv_v_dn = gk_price(spot, strike, T, vol - dv, rd, rf, is_call)
    volga = (pv_v_up - 2 * base + pv_v_dn) / dv ** 2 * dv  # per 1% move

    return FXOptionResult(
        price=base * notional, delta=delta, gamma=gamma,
        vega=vega * notional, theta=theta * notional,
        rho_dom=rho_dom * notional, rho_for=rho_for * notional,
        vanna=vanna, volga=volga, vol=vol, strike=strike,
    )


def get_sabr_vol(
    strike: float, sabr_params: dict, T: float,
) -> float:
    """Get SABR implied vol for a given strike."""
    F = sabr_params["forward"]
    try:
        sabr = Hagan2002LognormalSABR(F, 0.0, T)
        return sabr.lognormalvol(
            strike, T, sabr_params["alpha"], sabr_params["beta"],
            sabr_params["rho"], sabr_params["nu"],
        )
    except Exception:
        return sabr_params["alpha"]


def _strike_from_delta(delta_target: float, spot: float, T: float, rd: float, rf: float, vol: float, is_call: bool) -> float:
    """Invert GK delta to find strike."""
    F = spot * np.exp((rd - rf) * T)

    def delta_func(K):
        d1 = (np.log(spot / K) + (rd - rf + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
        if is_call:
            return np.exp(-rf * T) * norm.cdf(d1) - abs(delta_target)
        else:
            return np.exp(-rf * T) * (norm.cdf(d1) - 1) - delta_target

    try:
        return brentq(delta_func, F * 0.5, F * 2.0)
    except Exception:
        # Fallback: approximate
        d1_target = norm.ppf(abs(delta_target) * np.exp(rf * T))
        if not is_call:
            d1_target = -d1_target
        return F * np.exp(-d1_target * vol * np.sqrt(T) + 0.5 * vol ** 2 * T)
