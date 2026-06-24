"""Equity option pricing via Local Volatility (Dupire).

PRODUCTION PRACTICE:
- Desks use local vol surfaces for exotic pricing and hedging
- Implied vol surface = what you observe in the market (model-free)
- Local vol surface = instantaneous vol as function of (S, t), derived from implied via Dupire's formula
- Black-Scholes assumes flat vol → misprices OTM options and produces wrong hedge ratios
- Local vol matches ALL vanilla prices by construction but has known dynamics issues
  (vol moves opposite to spot, unlike reality). Stochastic-local-vol (SLV) fixes this
  but is significantly more complex.

We use QuantLib's BlackVarianceSurface → LocalVolSurface pipeline.
"""

import QuantLib as ql
import numpy as np
import pandas as pd
from datetime import date, timedelta


def build_vol_surface(
    option_chain: pd.DataFrame, spot: float, rate: float = 0.04,
    reference_date: date | None = None,
) -> tuple[ql.BlackVarianceSurface, ql.LocalVolSurface]:
    """Build implied and local vol surfaces from an option chain.

    Args:
        option_chain: DataFrame with columns [strike, expiry, iv, option_type]
        spot: current spot price
        rate: risk-free rate (flat assumption — production would use term structure)
        reference_date: valuation date (defaults to today)

    Returns:
        (implied_vol_surface, local_vol_surface)
    """
    ref = reference_date or date.today()
    ql_date = ql.Date(ref.day, ref.month, ref.year)
    ql.Settings.instance().evaluationDate = ql_date
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    day_count = ql.Actual365Fixed()

    # Filter to calls only for surface construction (avoid put-call parity noise)
    calls = option_chain[option_chain["option_type"] == "call"].copy()
    if calls.empty:
        calls = option_chain.copy()

    # Build strike × expiry grid
    expiries = sorted(calls["expiry"].unique())
    strikes = sorted(calls["strike"].unique())

    # Filter strikes to reasonable range around spot (avoid deep OTM noise)
    strikes = [k for k in strikes if 0.5 * spot <= k <= 2.0 * spot]
    if len(strikes) < 5:
        strikes = sorted(calls["strike"].unique())[:20]

    # Build vol matrix
    ql_dates = []
    for exp in expiries:
        exp_date = pd.Timestamp(exp).date()
        if exp_date <= ref:
            continue
        ql_dates.append(ql.Date(exp_date.day, exp_date.month, exp_date.year))

    if len(ql_dates) < 2:
        raise ValueError("Need at least 2 future expiries to build surface")

    vol_matrix = ql.Matrix(len(strikes), len(ql_dates))
    for j, exp in enumerate(expiries):
        exp_date = pd.Timestamp(exp).date()
        if exp_date <= ref:
            continue
        j_idx = next((i for i, d in enumerate(ql_dates) if d == ql.Date(exp_date.day, exp_date.month, exp_date.year)), None)
        if j_idx is None:
            continue
        exp_data = calls[calls["expiry"] == exp]
        for i, k in enumerate(strikes):
            row = exp_data[exp_data["strike"] == k]
            if not row.empty:
                vol_matrix[i][j_idx] = float(row["iv"].iloc[0])
            else:
                # Interpolate from nearest strikes
                nearest = exp_data.iloc[(exp_data["strike"] - k).abs().argsort()[:2]]
                if not nearest.empty:
                    vol_matrix[i][j_idx] = float(nearest["iv"].mean())
                else:
                    vol_matrix[i][j_idx] = 0.20  # fallback

    # Fill any zeros with neighbor average
    for i in range(vol_matrix.rows()):
        for j in range(vol_matrix.columns()):
            if vol_matrix[i][j] <= 0.01:
                vol_matrix[i][j] = 0.20

    implied_surface = ql.BlackVarianceSurface(
        ql_date, calendar,
        ql_dates, strikes, vol_matrix, day_count,
    )
    implied_surface.setInterpolation("bicubic")
    implied_surface.enableExtrapolation()

    # Build local vol surface from implied
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_date, rate, day_count)
    )
    div_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(ql_date, 0.015, day_count)  # ~1.5% div yield typical for SPY
    )
    vol_handle = ql.BlackVolTermStructureHandle(implied_surface)

    local_surface = ql.LocalVolSurface(
        vol_handle, rate_handle, div_handle, spot_handle,
    )
    local_surface.enableExtrapolation()

    return implied_surface, local_surface


def price_option(
    spot: float, strike: float, expiry_years: float, option_type: str,
    implied_vol_surface: ql.BlackVarianceSurface,
    rate: float = 0.04, div_yield: float = 0.015,
    reference_date: date | None = None,
) -> float:
    """Price a European option using the implied vol surface (local vol dynamics).

    NOTE: For vanilla pricing, using the implied vol from the surface with BS formula
    gives the correct market price by construction. The local vol model's value shows
    in exotic pricing and Greek computation (especially Vanna/Volga).
    """
    ref = reference_date or date.today()
    ql_date = ql.Date(ref.day, ref.month, ref.year)
    ql.Settings.instance().evaluationDate = ql_date
    day_count = ql.Actual365Fixed()

    maturity = ql_date + int(expiry_years * 365)
    vol = implied_vol_surface.blackVol(expiry_years, strike)

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
    return option.NPV()


def get_local_vol(local_surface: ql.LocalVolSurface, strike: float, time: float) -> float:
    """Query local vol at a specific (strike, time) point."""
    return local_surface.localVol(time, strike)
