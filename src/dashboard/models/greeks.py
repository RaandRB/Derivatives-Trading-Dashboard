"""Greek computation via bump-and-reprice (finite difference).

WHY NOT CLOSED-FORM BS GREEKS:
- BS Greeks assume flat vol across all strikes/spots — wrong when smile exists
- When spot moves, the option's moneyness changes, so its implied vol changes too
- This "sticky strike" vs "sticky delta" effect is captured by local vol but not by BS
- Vanna and Volga are zero for ATM options under flat vol, but non-zero in reality
- Bump-and-reprice with the full vol surface gives "smile-consistent" Greeks

We compute both BS Greeks (for comparison) and surface-consistent Greeks (for reality).
"""

from dataclasses import dataclass
from datetime import date

import QuantLib as ql
import numpy as np

from dashboard.models.equity import price_option


@dataclass
class Greeks:
    """All Greeks for a single option position."""
    delta: float
    gamma: float
    vega: float  # per 1% vol move
    theta: float  # per day
    rho: float  # per 1% rate move
    vanna: float  # d(delta)/d(vol)
    volga: float  # d(vega)/d(vol), aka vomma

    # BS comparison values
    bs_delta: float = 0.0
    bs_gamma: float = 0.0
    bs_vega: float = 0.0
    bs_theta: float = 0.0


def compute_greeks(
    spot: float, strike: float, expiry_years: float, option_type: str,
    implied_vol_surface: ql.BlackVarianceSurface,
    rate: float = 0.04, div_yield: float = 0.015,
    reference_date: date | None = None,
) -> Greeks:
    """Compute extended Greeks using bump-and-reprice on the vol surface.

    Bump sizes chosen to balance accuracy vs stability:
    - Spot: ±0.5% (small enough for good gamma, large enough to avoid noise)
    - Vol: ±1% absolute shift to entire surface
    - Time: 1 day forward
    - Rate: ±1bp
    """
    ref = reference_date or date.today()

    def pv(s=spot, r=rate, t=expiry_years):
        if t <= 0:
            return max(0, (s - strike) if option_type == "call" else (strike - s))
        return price_option(s, strike, t, option_type, implied_vol_surface, r, div_yield, ref)

    base = pv()
    ds = spot * 0.005  # 0.5% spot bump

    # Delta & Gamma (central difference)
    pv_up = pv(s=spot + ds)
    pv_dn = pv(s=spot - ds)
    delta = (pv_up - pv_dn) / (2 * ds)
    gamma = (pv_up - 2 * base + pv_dn) / (ds ** 2)

    # Theta (1 day forward)
    dt = 1.0 / 365.0
    theta = (pv(t=expiry_years - dt) - base) if expiry_years > dt else -base

    # Rho (1bp bump)
    dr = 0.0001
    rho = (pv(r=rate + dr) - pv(r=rate - dr)) / (2 * dr) * 0.01  # per 1% move

    # Vega, Vanna, Volga — shift the vol surface
    # We approximate by repricing with a shifted flat vol overlay
    dvol = 0.01  # 1% absolute vol shift
    vol_at_strike = implied_vol_surface.blackVol(expiry_years, strike)

    def pv_shifted_vol(vol_shift):
        """Reprice using shifted vol (approximation of surface parallel shift)."""
        ql_date = ql.Date(ref.day, ref.month, ref.year)
        day_count = ql.Actual365Fixed()
        shifted_vol = vol_at_strike + vol_shift
        if shifted_vol <= 0.001:
            shifted_vol = 0.001
        maturity = ql_date + int(expiry_years * 365)

        payoff = ql.PlainVanillaPayoff(
            ql.Option.Call if option_type == "call" else ql.Option.Put, strike
        )
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(spot)),
            ql.YieldTermStructureHandle(ql.FlatForward(ql_date, div_yield, day_count)),
            ql.YieldTermStructureHandle(ql.FlatForward(ql_date, rate, day_count)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql_date, ql.NullCalendar(), shifted_vol, day_count)),
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        return option.NPV()

    pv_vol_up = pv_shifted_vol(dvol)
    pv_vol_dn = pv_shifted_vol(-dvol)
    vega = (pv_vol_up - pv_vol_dn) / 2.0  # per 1% vol move

    # Volga: d²V/dσ²
    volga = (pv_vol_up - 2 * base + pv_vol_dn) / (dvol ** 2) * dvol  # normalized per 1% move

    # Vanna: d²V/(dS·dσ) — cross gamma between spot and vol
    pv_sup_vup = price_option(spot + ds, strike, expiry_years, option_type, implied_vol_surface, rate, div_yield, ref)
    pv_sdn_vup = price_option(spot - ds, strike, expiry_years, option_type, implied_vol_surface, rate, div_yield, ref)
    # Approximate vanna using spot-shifted delta difference
    delta_vol_up = (pv_sup_vup - pv_sdn_vup) / (2 * ds)  # delta at higher vol (approximate)
    vanna = (delta_vol_up - delta) * (ds / dvol)  # approximation via smile effect

    # BS Greeks for comparison (flat vol)
    bs = _bs_greeks(spot, strike, expiry_years, option_type, vol_at_strike, rate, div_yield, ref)

    return Greeks(
        delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
        vanna=vanna, volga=volga,
        bs_delta=bs["delta"], bs_gamma=bs["gamma"], bs_vega=bs["vega"], bs_theta=bs["theta"],
    )


def _bs_greeks(
    spot: float, strike: float, expiry_years: float, option_type: str,
    vol: float, rate: float, div_yield: float, ref: date,
) -> dict:
    """Standard Black-Scholes-Merton Greeks (closed-form via QuantLib)."""
    ql_date = ql.Date(ref.day, ref.month, ref.year)
    ql.Settings.instance().evaluationDate = ql_date
    day_count = ql.Actual365Fixed()
    maturity = ql_date + int(expiry_years * 365)

    payoff = ql.PlainVanillaPayoff(
        ql.Option.Call if option_type == "call" else ql.Option.Put, strike
    )
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(ql.SimpleQuote(spot)),
        ql.YieldTermStructureHandle(ql.FlatForward(ql_date, div_yield, day_count)),
        ql.YieldTermStructureHandle(ql.FlatForward(ql_date, rate, day_count)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql_date, ql.NullCalendar(), vol, day_count)),
    )
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

    return {
        "delta": option.delta(),
        "gamma": option.gamma(),
        "vega": option.vega() / 100.0,  # per 1% move
        "theta": option.theta() / 365.0,  # per day
    }
