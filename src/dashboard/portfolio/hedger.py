"""Hedge unwanted Greek exposures in a portfolio.

Given a portfolio with its Greek profile, construct a hedge overlay
that neutralizes specified Greeks using liquid instruments.
"""

import numpy as np
import cvxpy as cp

from dashboard.portfolio.builder import Portfolio, Instrument, GREEKS


def hedge_portfolio(
    portfolio: Portfolio,
    hedge_instruments: list[Instrument],
    greeks_to_hedge: list[str],
    tolerance: float = 0.05,
) -> tuple[Portfolio, dict]:
    """Compute a minimum-cost hedge overlay.

    Args:
        portfolio: existing portfolio to hedge
        hedge_instruments: liquid instruments available for hedging
        greeks_to_hedge: which Greeks to neutralize
        tolerance: acceptable residual as fraction of original exposure

    Returns:
        (hedge_portfolio, before_after_dict)
    """
    current_greeks = portfolio.total_greeks
    n = len(hedge_instruments)
    if n == 0:
        return Portfolio(), {"before": current_greeks, "after": current_greeks}

    # Greek matrix for hedge instruments
    G = np.zeros((n, len(GREEKS)))
    for i, inst in enumerate(hedge_instruments):
        for j, g in enumerate(GREEKS):
            G[i, j] = inst.greeks.get(g, 0.0)

    # Optimization: find hedge quantities
    # Use soft-constraint formulation — a real desk minimizes residual risk
    # weighted by priority, rather than demanding exact zeros (which may be
    # infeasible when Greeks are coupled, e.g. hedging delta without touching gamma).
    h = cp.Variable(n)

    # Weighted penalty on residual Greeks we want to hedge
    residual_penalty = 0
    for j, g in enumerate(GREEKS):
        if g in greeks_to_hedge:
            exposure = current_greeks[g]
            hedged = exposure + G[:, j] @ h
            # Normalize by exposure magnitude so each Greek contributes equally
            scale = max(abs(exposure), 0.5)
            residual_penalty += cp.abs(hedged) / scale

    # Minimize disturbance to Greeks we want to KEEP (the desired exposures)
    disturbance = 0
    for j, g in enumerate(GREEKS):
        if g not in greeks_to_hedge:
            exposure = current_greeks[g]
            scale = max(abs(exposure), 0.5)
            disturbance += cp.abs(G[:, j] @ h) / scale

    # Transaction cost proxy
    cost = cp.norm(h, 1)

    # Priority: minimize residual hedged Greeks >> preserve desired Greeks >> minimize cost
    objective = cp.Minimize(10.0 * residual_penalty + 1.0 * disturbance + 0.01 * cost)
    constraints = [cp.norm(h, "inf") <= 200]
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.ECOS, max_iters=5000)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, max_iters=10000)

    hedge_positions = []
    if h.value is not None:
        for i, qty in enumerate(h.value):
            if abs(qty) > 0.01:
                hedge_positions.append((hedge_instruments[i], float(qty)))

    hedge_pf = Portfolio(positions=hedge_positions)

    # Compute after-hedge Greeks
    after = {}
    hedge_greeks = hedge_pf.total_greeks
    for g in GREEKS:
        after[g] = current_greeks[g] + hedge_greeks[g]

    return hedge_pf, {"before": current_greeks, "after": after, "hedge_contribution": hedge_greeks}
