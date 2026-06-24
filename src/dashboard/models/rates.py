"""Interest rate swaption pricing with SABR vol model.

PRODUCTION PRACTICE:
- USD swaptions are quoted in normal (Bachelier) vol since ~2014 (negative rates era)
- The standard model is SABR with beta=0.5 (CEV backbone, market convention)
- Discounting uses OIS (SOFR for USD) — we approximate with Treasury curve
- Production systems calibrate SABR separately per expiry×tenor cell
- We calibrate alpha from ATM vol and use representative rho/nu

DISCREPANCY: We use Treasury rates as the swap curve. In reality, swap rates
are ~5-15bp higher than Treasuries due to credit/liquidity premium.
A production system would strip the swap curve from par swap rates directly.
"""

import QuantLib as ql
import numpy as np
from datetime import date

from pysabr import Hagan2002NormalSABR


def build_yield_curve(yield_data, reference_date: date | None = None) -> ql.YieldTermStructure:
    """Bootstrap a QuantLib yield curve from Treasury rates.

    Args:
        yield_data: DataFrame with columns [tenor_years, rate]
    """
    ref = reference_date or date.today()
    ql_date = ql.Date(ref.day, ref.month, ref.year)
    ql.Settings.instance().evaluationDate = ql_date
    day_count = ql.Actual365Fixed()

    dates = [ql_date]
    rates = [float(yield_data["rate"].iloc[0])]

    for _, row in yield_data.iterrows():
        d = ql_date + int(row["tenor_years"] * 365)
        dates.append(d)
        rates.append(float(row["rate"]))

    curve = ql.ZeroCurve(dates, rates, day_count)
    curve.enableExtrapolation()
    return curve


def calibrate_sabr(atm_vol_bp: float, forward: float, sabr_params: dict) -> dict:
    """Calibrate SABR alpha from ATM normal vol, given beta/rho/nu.

    Uses the SABR normal vol approximation (Hagan 2002).
    Alpha is the free parameter calibrated to match ATM vol.

    Args:
        atm_vol_bp: ATM normal vol in basis points (e.g., 85 = 85bp/year)
        forward: forward swap rate (decimal, e.g., 0.04)
        sabr_params: dict with keys beta, rho, nu
    """
    beta = sabr_params["beta"]
    rho = sabr_params["rho"]
    nu = sabr_params["nu"]
    atm_vol = atm_vol_bp / 10000.0  # convert bp to decimal

    # pysabr calibration: find alpha that matches ATM vol
    # For normal SABR, alpha ≈ atm_normal_vol / f^(1-beta) as a starting point
    alpha = atm_vol / (forward ** (1 - beta)) if forward > 0 else atm_vol

    # Refine with pysabr
    try:
        sabr = Hagan2002NormalSABR(forward, 0.0, 1.0)  # shift=0, T=1 placeholder
        # Simple Newton to match ATM
        for _ in range(20):
            model_vol = sabr.normalvol(forward, 1.0, alpha, beta, rho, nu)
            if abs(model_vol - atm_vol) < 1e-8:
                break
            # Bump alpha
            model_vol_up = sabr.normalvol(forward, 1.0, alpha * 1.001, beta, rho, nu)
            dalpha = (model_vol_up - model_vol) / (alpha * 0.001)
            if abs(dalpha) < 1e-12:
                break
            alpha -= (model_vol - atm_vol) / dalpha
            alpha = max(alpha, 1e-6)
    except Exception:
        pass  # keep initial estimate

    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": nu}


def price_swaption(
    expiry_years: float, tenor_years: float, strike: float | None,
    payer: bool, notional: float,
    yield_curve_data, swaption_vols: dict,
    reference_date: date | None = None,
) -> dict:
    """Price a European swaption using SABR normal vol.

    Args:
        expiry_years: option expiry in years
        tenor_years: underlying swap tenor in years
        strike: swap rate strike (None = ATM forward)
        payer: True for payer swaption, False for receiver
        notional: notional amount
        yield_curve_data: DataFrame from data/rates.py
        swaption_vols: dict from data/rates.py

    Returns:
        dict with price, greeks, forward_rate, vol
    """
    ref = reference_date or date.today()
    curve = build_yield_curve(yield_curve_data, ref)

    # Compute forward swap rate (annuity-weighted average of forward rates)
    forward = _compute_forward_swap_rate(curve, expiry_years, tenor_years, ref)

    if strike is None:
        strike = forward

    # Get SABR params and calibrate
    atm_vols = swaption_vols.get("atm_normal_vols_bp", {})
    sabr_params = swaption_vols.get("sabr_params", {"beta": 0.5, "rho": -0.2, "nu": 0.3})

    # Find closest expiry/tenor in the vol cube
    exp_key = _closest_key(atm_vols, expiry_years)
    if exp_key and isinstance(atm_vols[exp_key], dict):
        tenor_key = _closest_key(atm_vols[exp_key], tenor_years)
        atm_vol_bp = atm_vols[exp_key].get(tenor_key, 80)
    else:
        atm_vol_bp = 80  # fallback

    calibrated = calibrate_sabr(float(atm_vol_bp), forward, sabr_params)

    # Compute vol at strike using calibrated SABR
    try:
        sabr = Hagan2002NormalSABR(forward, 0.0, expiry_years)
        vol = sabr.normalvol(strike, expiry_years, calibrated["alpha"], calibrated["beta"], calibrated["rho"], calibrated["nu"])
    except Exception:
        vol = atm_vol_bp / 10000.0

    # Bachelier (normal) swaption pricing
    annuity = _compute_annuity(curve, expiry_years, tenor_years, ref)
    price = _bachelier_price(forward, strike, vol, expiry_years, payer) * annuity * notional

    # Greeks via bump-and-reprice
    dr = 0.0001  # 1bp
    # Rho: bump curve
    price_rate_up = price * 1.0  # simplified — full reprice would rebuild curve
    # For educational purposes, approximate:
    dv01 = annuity * notional * 0.0001 * tenor_years  # approx DV01

    return {
        "price": price,
        "forward_rate": forward,
        "strike": strike,
        "normal_vol_bp": vol * 10000,
        "annuity": annuity,
        "delta": _bachelier_delta(forward, strike, vol, expiry_years, payer),
        "vega_bp": annuity * notional * _bachelier_vega(forward, strike, vol, expiry_years) / 10000,
        "dv01": dv01,
    }


def _compute_forward_swap_rate(curve, expiry_years, tenor_years, ref):
    """Approximate forward swap rate from zero curve."""
    ql_date = ql.Date(ref.day, ref.month, ref.year)
    day_count = ql.Actual365Fixed()

    t_start = expiry_years
    t_end = expiry_years + tenor_years

    # df(t) from curve
    df_start = curve.discount(t_start)
    df_end = curve.discount(t_end)

    # Approximate annuity (annual payments)
    annuity = sum(curve.discount(t_start + i) for i in range(1, int(tenor_years) + 1))
    if annuity <= 0:
        annuity = tenor_years * df_start

    forward = (df_start - df_end) / annuity
    return forward


def _compute_annuity(curve, expiry_years, tenor_years, ref):
    """PV01 / annuity factor."""
    annuity = sum(curve.discount(expiry_years + i) for i in range(1, int(tenor_years) + 1))
    return max(annuity, 0.01)


def _bachelier_price(forward, strike, vol, T, payer):
    """Bachelier (normal model) option price."""
    from scipy.stats import norm
    if vol <= 0 or T <= 0:
        intrinsic = (forward - strike) if payer else (strike - forward)
        return max(0, intrinsic)
    d = (forward - strike) / (vol * np.sqrt(T))
    sign = 1 if payer else -1
    return sign * (forward - strike) * norm.cdf(sign * d) + vol * np.sqrt(T) * norm.pdf(d)


def _bachelier_delta(forward, strike, vol, T, payer):
    """Bachelier delta."""
    from scipy.stats import norm
    if vol <= 0 or T <= 0:
        return 1.0 if (forward > strike) == payer else 0.0
    d = (forward - strike) / (vol * np.sqrt(T))
    return norm.cdf(d) if payer else norm.cdf(d) - 1


def _bachelier_vega(forward, strike, vol, T):
    """Bachelier vega (dV/d_sigma)."""
    from scipy.stats import norm
    if T <= 0:
        return 0.0
    return np.sqrt(T) * norm.pdf((forward - strike) / (vol * np.sqrt(T))) if vol > 0 else np.sqrt(T) * norm.pdf(0)


def _closest_key(d: dict, target_years: float) -> str | None:
    """Find closest tenor/expiry key in a dict."""
    if not d:
        return None
    mapping = {"1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "7Y": 7, "10Y": 10, "20Y": 20, "30Y": 30}
    best_key = None
    best_dist = float("inf")
    for k in d.keys():
        y = mapping.get(k, float("inf"))
        if abs(y - target_years) < best_dist:
            best_dist = abs(y - target_years)
            best_key = k
    return best_key
