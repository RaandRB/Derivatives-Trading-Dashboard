"""Core tests for portfolio builder, hedger, and simulation."""

import numpy as np
import pandas as pd
from datetime import date, timedelta

from dashboard.portfolio.builder import Portfolio, Instrument, build_portfolio, GREEKS
from dashboard.portfolio.hedger import hedge_portfolio
from dashboard.simulation.engine import simulate_forward
from dashboard.models.fx import price_fx_option, delta_space_to_strikes
from dashboard.data.rates import get_yield_curve, get_swaption_vols


def _sample_instruments():
    return [
        Instrument("Call", "equity", "call", {"delta": 0.52, "gamma": 0.008, "vega": 0.35, "theta": -0.15, "rho": 0.12, "vanna": 0.01, "volga": 0.02}),
        Instrument("Put", "equity", "put", {"delta": -0.48, "gamma": 0.008, "vega": 0.35, "theta": -0.14, "rho": -0.10, "vanna": -0.01, "volga": 0.02}),
        Instrument("Stock", "equity", "underlying", {"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0}),
    ]


def test_portfolio_long_gamma_neutral_delta():
    instruments = _sample_instruments()
    target = {"delta": "neutral", "gamma": "long", "vega": "neutral", "theta": "neutral", "rho": "neutral", "vanna": "neutral", "volga": "neutral"}
    pf = build_portfolio(instruments, target)
    greeks = pf.total_greeks
    assert greeks["gamma"] > 0, f"Expected positive gamma, got {greeks['gamma']}"
    assert abs(greeks["delta"]) < 1.0, f"Expected near-zero delta, got {greeks['delta']}"


def test_portfolio_long_vega():
    instruments = _sample_instruments()
    target = {"delta": "neutral", "gamma": "neutral", "vega": "long", "theta": "neutral", "rho": "neutral", "vanna": "neutral", "volga": "neutral"}
    pf = build_portfolio(instruments, target)
    greeks = pf.total_greeks
    assert greeks["vega"] > 0, f"Expected positive vega, got {greeks['vega']}"


def test_hedger_reduces_delta():
    instruments = _sample_instruments()
    # Build a portfolio with delta exposure
    pf = Portfolio(positions=[(instruments[0], 10.0)])  # 10 calls → delta ≈ 5.2
    hedge, ba = hedge_portfolio(pf, instruments, ["delta"])
    assert abs(ba["after"]["delta"]) < abs(ba["before"]["delta"]) * 0.5


def test_simulation_runs():
    instruments = _sample_instruments()
    pf = Portfolio(positions=[(instruments[0], 5.0), (instruments[2], -2.6)])
    result = simulate_forward(pf, spot=550.0, n_days=10, seed=123)
    assert len(result.steps) == 10
    # P&L attribution sums should roughly equal total
    for step in result.steps:
        attr_sum = sum(step.pnl_attribution.values())
        assert abs(attr_sum + step.residual - step.pnl_total) < 1e-10


def test_fx_option_call_put_parity():
    """Call - Put ≈ S*exp(-rf*T) - K*exp(-rd*T) (Garman-Kohlhagen parity)."""
    spot, strike, T, vol, rd, rf = 1.08, 1.10, 0.5, 0.08, 0.04, 0.03
    call = price_fx_option(spot, strike, T, True, vol, rd, rf, notional=1.0)
    put = price_fx_option(spot, strike, T, False, vol, rd, rf, notional=1.0)
    parity = spot * np.exp(-rf * T) - strike * np.exp(-rd * T)
    assert abs(call.price - put.price - parity) < 1e-6, f"Parity violated: {call.price - put.price} vs {parity}"


def test_fx_delta_bounds():
    result = price_fx_option(1.08, 1.08, 0.25, True, 0.08)
    assert 0 < result.delta < 1, f"Call delta out of bounds: {result.delta}"


def test_yield_curve_monotonic_discount():
    curve = get_yield_curve()
    assert len(curve) >= 5
    # Rates should be positive
    assert (curve["rate"] > 0).all()


def test_swaption_vols_structure():
    vols = get_swaption_vols()
    assert "atm_normal_vols_bp" in vols
    assert "sabr_params" in vols
    assert vols["sabr_params"]["beta"] == 0.5


def test_delta_space_strikes_ordered():
    """25D put strike < ATM strike < 25D call strike."""
    strikes = delta_space_to_strikes(1.08, 0.08, -0.01, 0.005, 0.04, 0.03, 0.25)
    assert strikes["p25_strike"] < strikes["atm_strike"] < strikes["c25_strike"]
