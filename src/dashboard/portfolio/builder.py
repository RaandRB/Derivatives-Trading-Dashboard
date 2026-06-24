"""Portfolio construction from target Greek exposures.

Given a user-specified risk profile (which Greeks to be long/short/neutral),
selects instruments and quantities to achieve that profile using optimization.
"""

from dataclasses import dataclass, field
import numpy as np
import cvxpy as cp


GREEKS = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]


@dataclass
class Instrument:
    """An available tradeable instrument with its Greeks."""
    name: str
    asset_class: str  # "equity", "fx", "rates"
    instrument_type: str  # "call", "put", "swap", "swaption", "underlying"
    greeks: dict[str, float]  # {greek_name: value_per_unit}
    price: float = 0.0
    description: str = ""


@dataclass
class Portfolio:
    """A portfolio of instruments with quantities."""
    positions: list[tuple[Instrument, float]] = field(default_factory=list)  # (instrument, quantity)

    @property
    def total_greeks(self) -> dict[str, float]:
        totals = {g: 0.0 for g in GREEKS}
        for inst, qty in self.positions:
            for g in GREEKS:
                totals[g] += inst.greeks.get(g, 0.0) * qty
        return totals

    def greeks_table(self) -> list[dict]:
        """Per-position Greek breakdown for display."""
        rows = []
        for inst, qty in self.positions:
            row = {"instrument": inst.name, "quantity": qty, "asset_class": inst.asset_class}
            for g in GREEKS:
                row[g] = inst.greeks.get(g, 0.0) * qty
            rows.append(row)
        return rows


def build_portfolio(
    instruments: list[Instrument],
    target_greeks: dict[str, str],  # {greek: "long"/"short"/"neutral"}
    target_magnitude: float = 1.0,
    max_positions: int = 5,
) -> Portfolio:
    """Construct a portfolio that expresses the desired view — nothing more.

    Only maximizes the Greeks the user wants exposure to.
    Does NOT attempt to neutralize unwanted Greeks — that's the hedger's job.
    This separation makes the educational flow clear:
    "raw view expression" → "hedge the rest" → "see residual"

    Args:
        instruments: available instruments with their Greeks
        target_greeks: which Greeks to be long/short/neutral
        target_magnitude: scale factor for desired exposures
        max_positions: max distinct instruments (keep it simple/readable)
    """
    n = len(instruments)
    if n == 0:
        return Portfolio()

    # Build Greek matrix: rows = instruments, cols = greeks
    G = np.zeros((n, len(GREEKS)))
    for i, inst in enumerate(instruments):
        for j, g in enumerate(GREEKS):
            G[i, j] = inst.greeks.get(g, 0.0)

    w = cp.Variable(n)

    # Objective: ONLY maximize desired Greek exposures
    objective_terms = []
    constraints = []

    for j, g in enumerate(GREEKS):
        direction = target_greeks.get(g, "neutral")
        greek_exposure = G[:, j] @ w

        if direction == "long":
            objective_terms.append(greek_exposure)
            constraints.append(greek_exposure >= target_magnitude * 0.1)
        elif direction == "short":
            objective_terms.append(-greek_exposure)
            constraints.append(greek_exposure <= -target_magnitude * 0.1)
        # "neutral" Greeks are IGNORED here — deliberately left unhedged

    # Regularization: prefer fewer, smaller positions for readability
    objective_terms.append(-0.1 * cp.norm(w, 1))

    objective = cp.Maximize(sum(objective_terms))
    constraints.append(cp.norm(w, "inf") <= 50)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.ECOS, max_iters=5000)
    except cp.SolverError:
        prob.solve(solver=cp.SCS)

    if w.value is None:
        prob2 = cp.Problem(cp.Maximize(sum(objective_terms)), [cp.norm(w, "inf") <= 50])
        prob2.solve(solver=cp.SCS)

    if w.value is None:
        return Portfolio()

    quantities = w.value
    indices = np.argsort(-np.abs(quantities))[:max_positions]

    positions = []
    for i in indices:
        if abs(quantities[i]) > 0.01:
            positions.append((instruments[i], float(quantities[i])))

    return Portfolio(positions=positions)
