"""Forward simulation engine for dynamic hedge P&L attribution.

Steps the portfolio forward in time, applies market moves (spot, vol, rates),
and decomposes P&L into Greek contributions using Taylor expansion:

ΔP&L ≈ Delta·ΔS + ½·Gamma·ΔS² + Vega·Δσ + Theta·Δt + Rho·Δr
      + Vanna·ΔS·Δσ + ½·Volga·Δσ²

The residual (actual reprice - Taylor approximation) shows higher-order effects.
"""

from dataclasses import dataclass, field
import numpy as np

from dashboard.portfolio.builder import Portfolio, GREEKS


@dataclass
class SimStep:
    """One time step in the simulation."""
    day: int
    spot_change_pct: float
    vol_change_pct: float  # parallel shift in absolute terms
    rate_change_bp: float
    pnl_total: float
    pnl_attribution: dict[str, float]  # Greek → P&L contribution
    residual: float
    cumulative_pnl: float
    greeks_after: dict[str, float]
    spot_level: float = 0.0
    vol_level: float = 0.0  # cumulative vol shift from start
    rate_level_bp: float = 0.0  # cumulative rate shift from start


@dataclass
class SimulationResult:
    """Full simulation output."""
    steps: list[SimStep] = field(default_factory=list)
    initial_greeks: dict[str, float] = field(default_factory=dict)


def simulate_forward(
    portfolio: Portfolio,
    spot: float,
    n_days: int = 5,
    spot_vol: float = 0.01,
    vol_of_vol: float = 0.005,
    rate_vol_bp: float = 2.0,
    seed: int | None = None,
    rebalance_every: int | None = None,
    rebalance_cost_bps: float = 5.0,
    use_historical: bool = False,
) -> SimulationResult:
    """Simulate portfolio P&L over multiple days.

    Uses historical-caliber random draws for market moves.
    P&L attribution via 2nd order Taylor expansion of portfolio value.

    Args:
        portfolio: current portfolio with positions
        spot: initial spot price
        n_days: number of days to simulate
        spot_vol: daily spot return standard deviation (ignored if use_historical)
        vol_of_vol: daily absolute vol change std dev
        rate_vol_bp: daily rate change std dev in basis points
        seed: random seed for reproducibility
        rebalance_every: reset delta/gamma/vanna Greeks every N days (simulates
            re-hedging). Theta/vega are NOT reset — you still own the same options.
        rebalance_cost_bps: cost per rebalance in bps of notional (bid-ask + impact)
        use_historical: if True, sample spot moves from real SPX history
    """
    rng = np.random.default_rng(seed)
    initial_greeks = portfolio.total_greeks
    greeks = initial_greeks.copy()
    result = SimulationResult(initial_greeks=initial_greeks.copy())

    # Get historical returns if requested
    historical_returns = None
    if use_historical:
        from dashboard.data.historical import sample_historical_moves
        historical_returns = sample_historical_moves(n_days, seed=seed)

    cumulative = 0.0
    current_spot = spot
    cum_vol_shift = 0.0
    cum_rate_shift_bp = 0.0

    for day in range(1, n_days + 1):
        # Rebalance: reset spot-sensitive Greeks (delta hedge adjustment)
        # Theta and vega are NOT reset — you still own the same options, time still passes.
        rebalance_cost = 0.0
        if rebalance_every and day > 1 and (day - 1) % rebalance_every == 0:
            # Cost proportional to how much delta drifted (= how much you need to trade)
            delta_drift = abs(greeks["delta"] - initial_greeks["delta"])
            rebalance_cost = delta_drift * current_spot * rebalance_cost_bps / 10000.0
            # Only reset delta (the thing you actually rebalance with stock/futures)
            greeks["delta"] = initial_greeks["delta"]
            greeks["gamma"] = initial_greeks["gamma"]
            greeks["vanna"] = initial_greeks["vanna"]
            # theta, vega, rho, volga continue to evolve — they aren't free to reset

        # Stochastic realized vol: daily vol itself varies (mean-reverting around spot_vol)
        # This is the key risk for gamma traders: some days are dead calm, some aren't.
        if historical_returns is not None:
            dS_pct = float(historical_returns[day - 1])
        else:
            daily_realized_vol = abs(rng.normal(spot_vol, spot_vol * 0.5))
            dS_pct = rng.normal(0, daily_realized_vol)
        dS = current_spot * dS_pct
        dvol = rng.normal(0, vol_of_vol)
        dr_bp = rng.normal(0, rate_vol_bp)
        dr = dr_bp / 10000.0
        dt = 1.0 / 365.0

        # P&L attribution (Taylor expansion)
        pnl_delta = greeks["delta"] * dS
        pnl_gamma = 0.5 * greeks["gamma"] * dS ** 2
        pnl_vega = greeks["vega"] * dvol * 100  # vega is per 1%, dvol is absolute
        pnl_theta = greeks["theta"]  # already per day
        pnl_rho = greeks["rho"] * dr * 100  # rho is per 1%
        pnl_vanna = greeks["vanna"] * dS * dvol
        pnl_volga = 0.5 * greeks["volga"] * (dvol * 100) ** 2

        pnl_total = pnl_delta + pnl_gamma + pnl_vega + pnl_theta + pnl_rho + pnl_vanna + pnl_volga
        # In a real system, you'd reprice the full portfolio and compute residual
        # Here residual is simulated as a small random noise
        residual = rng.normal(0, abs(pnl_total) * 0.05) if abs(pnl_total) > 0 else 0
        pnl_total += residual - rebalance_cost

        cumulative += pnl_total
        current_spot += dS
        cum_vol_shift += dvol * 100
        cum_rate_shift_bp += dr_bp

        # Update greeks approximately (gamma changes delta, volga changes vega)
        greeks = {
            "delta": greeks["delta"] + greeks["gamma"] * dS + greeks["vanna"] * dvol,
            "gamma": greeks["gamma"] * (1 - abs(dS_pct) * 0.1),  # gamma decays
            "vega": greeks["vega"] + greeks["volga"] * dvol,
            "theta": greeks["theta"] * (1 + dt),  # theta accelerates near expiry
            "rho": greeks["rho"],
            "vanna": greeks["vanna"] * (1 - abs(dS_pct) * 0.05),
            "volga": greeks["volga"] * (1 - abs(dvol) * 0.1),
        }

        result.steps.append(SimStep(
            day=day,
            spot_change_pct=dS_pct * 100,
            vol_change_pct=dvol * 100,
            rate_change_bp=dr_bp,
            pnl_total=pnl_total,
            pnl_attribution={
                "delta": pnl_delta, "gamma": pnl_gamma, "vega": pnl_vega,
                "theta": pnl_theta, "rho": pnl_rho, "vanna": pnl_vanna, "volga": pnl_volga,
            },
            residual=residual,
            cumulative_pnl=cumulative,
            greeks_after=greeks.copy(),
            spot_level=current_spot,
            vol_level=cum_vol_shift,
            rate_level_bp=cum_rate_shift_bp,
        ))

    return result
