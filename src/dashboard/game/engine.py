"""Trading Game — practice portfolio management under realistic constraints.

The player manages a derivatives book through a historical period, making
decisions each "round" (week). Features a dynamic options market where Greeks
update realistically as spot moves and time passes.
"""

import numpy as np
from dataclasses import dataclass, field
from scipy.stats import norm


GREEKS = ["delta", "gamma", "vega", "theta", "rho", "vanna", "volga"]

CONSTRAINTS = {
    "max_drawdown_pct": 15.0,
    "max_delta": 50.0,
    "max_vega": 30.0,
    "max_position_units": 100,
    "starting_capital": 10000.0,
}


# --- Black-Scholes Greeks Engine ---

def _bs_greeks(spot: float, strike: float, tau: float, vol: float, r: float, opt_type: str) -> dict:
    """Compute all Greeks for a European option via Black-Scholes.

    Args:
        spot: current underlying price
        strike: option strike
        tau: time to expiry in years (must be > 0)
        vol: implied volatility (annualized)
        r: risk-free rate (annualized)
        opt_type: "call" or "put"

    Returns dict of greeks per unit notional (1 contract = 1 share underlying).
    """
    if tau <= 0:
        # Expired
        intrinsic = max(spot - strike, 0) if opt_type == "call" else max(strike - spot, 0)
        return {"delta": 0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0, "price": intrinsic}

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(spot / strike) + (r + 0.5 * vol**2) * tau) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    npd1 = norm.pdf(d1)
    df = np.exp(-r * tau)

    if opt_type == "call":
        price = spot * nd1 - strike * df * nd2
        delta = nd1
        rho = strike * tau * df * nd2 / 100  # per 1% rate move
    else:
        price = strike * df * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = nd1 - 1.0
        rho = -strike * tau * df * norm.cdf(-d2) / 100

    gamma = npd1 / (spot * vol * sqrt_tau)
    vega = spot * npd1 * sqrt_tau / 100  # per 1% vol move
    theta = (-(spot * npd1 * vol) / (2 * sqrt_tau) - r * strike * df * (nd2 if opt_type == "call" else norm.cdf(-d2))) / 365  # per day
    vanna = -npd1 * d2 / (vol * spot) * (spot / 100)  # ∂delta/∂vol per 1% vol
    volga = vega * d1 * d2 / vol  # ∂vega/∂vol

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
        "vanna": float(vanna),
        "volga": float(volga),
        "price": float(price),
    }


# --- Option Contract ---

@dataclass
class OptionContract:
    """A specific tradeable option contract."""
    contract_id: str
    opt_type: str  # "call", "put", "stock"
    strike: float  # 0 for stock
    expiry_days: int  # days to expiry at inception
    vol: float  # implied vol at trade time
    inception_round: int = 0

    def label(self) -> str:
        if self.opt_type == "stock":
            return "Stock"
        moneyness = self.strike
        return f"{self.opt_type.capitalize()} K={moneyness:.1f} ({self.expiry_days}d)"

    def greeks_at(self, spot: float, days_elapsed: int, vol: float, r: float = 0.04) -> dict:
        """Compute current Greeks given market state."""
        if self.opt_type == "stock":
            return {"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0, "price": spot}
        tau = max(self.expiry_days - days_elapsed, 0) / 365.0
        return _bs_greeks(spot, self.strike, tau, vol, r, self.opt_type)

    def days_remaining(self, days_elapsed: int) -> int:
        return max(self.expiry_days - days_elapsed, 0)


# --- Market Generator ---

def generate_market(spot: float, days_elapsed: int, vol: float, r: float = 0.04) -> list[dict]:
    """Generate the current option market — what's available to trade.

    Produces a realistic set of options at various strikes and expiries,
    with computed Greeks and bid/ask prices.
    """
    market = []

    # Stock always available
    market.append({
        "contract": OptionContract("stock", "stock", 0, 9999, 0),
        "greeks": {"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0, "price": spot},
        "bid_ask_spread": spot * 0.0002,  # ~2bps for stock
    })

    # Option expiries: 1 week, 1 month, 3 months
    expiry_configs = [
        (7, "1W"),
        (30, "1M"),
        (90, "3M"),
    ]

    # Strikes: 90%, 95%, ATM, 105%, 110% of current spot
    strike_pcts = [0.90, 0.95, 1.00, 1.05, 1.10]

    for exp_days, exp_label in expiry_configs:
        remaining = exp_days  # these are new contracts available now
        tau = remaining / 365.0
        if tau <= 0:
            continue

        for strike_pct in strike_pcts:
            strike = round(spot * strike_pct, 2)
            for opt_type in ["call", "put"]:
                contract_id = f"{opt_type}_{strike:.1f}_{exp_label}_r{days_elapsed}"
                contract = OptionContract(
                    contract_id=contract_id,
                    opt_type=opt_type,
                    strike=strike,
                    expiry_days=exp_days + days_elapsed,  # absolute expiry day
                    vol=vol,
                    inception_round=days_elapsed // 5,
                )
                greeks = contract.greeks_at(spot, days_elapsed, vol, r)

                # Spread proportional to gamma (wider for harder to hedge)
                spread = greeks["price"] * 0.03 + abs(greeks["gamma"]) * spot * 0.5
                spread = max(spread, 0.01)

                market.append({
                    "contract": contract,
                    "greeks": greeks,
                    "bid_ask_spread": spread,
                    "label": f"{opt_type.upper()} K={strike:.0f} {exp_label}",
                })

    return market


# --- Game State ---

@dataclass
class Position:
    """A held position."""
    contract: OptionContract
    quantity: int
    entry_price: float
    entry_round: int


@dataclass
class GameState:
    """Tracks the full game state across rounds."""
    round: int = 0
    total_rounds: int = 20
    days_per_round: int = 5
    capital: float = 10000.0
    peak_capital: float = 10000.0
    positions: list = field(default_factory=list)  # list of Position
    pnl_history: list = field(default_factory=list)
    trade_history: list = field(default_factory=list)
    market_data: list = field(default_factory=list)
    vol_data: list = field(default_factory=list)  # implied vol path
    spot_path: list = field(default_factory=list)
    breached_limit: str | None = None
    use_historical: bool = True
    implied_vol: float = 0.16  # current implied vol level

    @property
    def days_elapsed(self) -> int:
        return self.round * self.days_per_round

    def portfolio_greeks(self) -> dict:
        """Compute total portfolio Greeks from all positions."""
        totals = {g: 0.0 for g in GREEKS}
        spot = self.spot_path[-1] if self.spot_path else 100.0
        for pos in self.positions:
            g = pos.contract.greeks_at(spot, self.days_elapsed, self.implied_vol)
            for greek in GREEKS:
                totals[greek] += g[greek] * pos.quantity
        return totals

    def portfolio_value(self) -> float:
        """Mark-to-market value of all positions."""
        spot = self.spot_path[-1] if self.spot_path else 100.0
        value = 0.0
        for pos in self.positions:
            g = pos.contract.greeks_at(spot, self.days_elapsed, self.implied_vol)
            value += g["price"] * pos.quantity
        return value

    def positions_detail(self) -> list[dict]:
        """Detailed position info for display."""
        spot = self.spot_path[-1] if self.spot_path else 100.0
        details = []
        for pos in self.positions:
            g = pos.contract.greeks_at(spot, self.days_elapsed, self.implied_vol)
            remaining = pos.contract.days_remaining(self.days_elapsed)
            details.append({
                "label": pos.contract.label(),
                "qty": pos.quantity,
                "days_left": remaining,
                "delta": g["delta"] * pos.quantity,
                "gamma": g["gamma"] * pos.quantity,
                "vega": g["vega"] * pos.quantity,
                "theta": g["theta"] * pos.quantity,
                "pnl": (g["price"] - pos.entry_price) * pos.quantity,
                "expired": remaining <= 0,
            })
        return details


# --- Game Logic ---

def init_game(total_rounds: int = 20, use_historical: bool = True, seed: int = 42) -> GameState:
    """Initialize a new game."""
    state = GameState(total_rounds=total_rounds, use_historical=use_historical)
    state.spot_path = [100.0]

    rng = np.random.default_rng(seed)
    total_days = total_rounds * state.days_per_round

    if use_historical:
        from dashboard.data.historical import get_spx_returns
        df = get_spx_returns()
        max_start = len(df) - total_days - 1
        start_idx = int(rng.integers(0, max(1, max_start)))
        returns = df["return_pct"].iloc[start_idx:start_idx + total_days].values
        # Derive vol from the data
        vol_20d = df["realized_vol_20d"].iloc[start_idx:start_idx + total_days].values
        # Convert daily std to annualized
        state.vol_data = (vol_20d * np.sqrt(252)).tolist()
    else:
        spot_vol = 0.012
        returns = np.array([
            rng.normal(0, abs(rng.normal(spot_vol, spot_vol * 0.5)))
            for _ in range(total_days)
        ])
        state.vol_data = [0.16] * total_days

    state.market_data = returns.tolist()
    state.implied_vol = state.vol_data[0] if state.vol_data else 0.16
    return state


def get_round_context(state: GameState) -> dict:
    """Get market context visible to the player."""
    days_elapsed = state.days_elapsed
    past_returns = state.market_data[:days_elapsed]
    spot = state.spot_path[-1]

    context = {
        "round": state.round + 1,
        "total_rounds": state.total_rounds,
        "spot": spot,
        "implied_vol": state.implied_vol * 100,
        "capital": state.capital,
        "pnl_so_far": state.capital - CONSTRAINTS["starting_capital"],
        "drawdown_pct": (1 - state.capital / state.peak_capital) * 100 if state.peak_capital > 0 else 0,
        "current_greeks": state.portfolio_greeks(),
        "positions": state.positions_detail(),
        "portfolio_value": state.portfolio_value(),
    }

    if len(past_returns) >= 5:
        recent = np.array(past_returns[-5:])
        context["recent_5d_return"] = float(recent.sum() * 100)
        context["recent_5d_vol"] = float(recent.std() * np.sqrt(252) * 100)
    else:
        context["recent_5d_return"] = 0.0
        context["recent_5d_vol"] = state.implied_vol * 100

    if len(past_returns) >= 20:
        recent20 = np.array(past_returns[-20:])
        context["recent_20d_vol"] = float(recent20.std() * np.sqrt(252) * 100)
        context["trend"] = "up" if recent20.sum() > 0.02 else ("down" if recent20.sum() < -0.02 else "flat")
    else:
        context["recent_20d_vol"] = state.implied_vol * 100
        context["trend"] = "unknown"

    return context


def get_market(state: GameState) -> list[dict]:
    """Get currently available options to trade."""
    spot = state.spot_path[-1]
    return generate_market(spot, state.days_elapsed, state.implied_vol)


def execute_trade(state: GameState, contract: OptionContract, quantity: int) -> list[str]:
    """Execute a single trade. Returns warnings."""
    warnings = []
    if quantity == 0:
        return warnings

    spot = state.spot_path[-1]
    greeks = contract.greeks_at(spot, state.days_elapsed, state.implied_vol)
    price = greeks["price"]

    # Find existing position in same contract (by id)
    existing = next((p for p in state.positions if p.contract.contract_id == contract.contract_id), None)
    new_qty = (existing.quantity if existing else 0) + quantity

    if abs(new_qty) > CONSTRAINTS["max_position_units"]:
        warnings.append(f"Position limit: would be {new_qty} (max ±{CONSTRAINTS['max_position_units']})")
        return warnings

    # Compute spread cost
    market_items = get_market(state)
    market_item = next((m for m in market_items if m["contract"].contract_id == contract.contract_id), None)
    spread = market_item["bid_ask_spread"] if market_item else price * 0.03
    cost = abs(quantity) * spread * 0.5  # half spread per side

    # Update position
    if existing:
        if new_qty == 0:
            state.positions.remove(existing)
        else:
            existing.quantity = new_qty
    else:
        state.positions.append(Position(
            contract=contract,
            quantity=quantity,
            entry_price=price,
            entry_round=state.round,
        ))

    state.capital -= cost
    state.trade_history.append({
        "round": state.round + 1,
        "trade": contract.label(),
        "qty": quantity,
        "price": price,
        "cost": cost,
    })

    # Check limits
    portfolio_greeks = state.portfolio_greeks()
    if abs(portfolio_greeks["delta"]) > CONSTRAINTS["max_delta"]:
        warnings.append(f"⚠️ Delta limit: {portfolio_greeks['delta']:.1f} (max ±{CONSTRAINTS['max_delta']})")
    if abs(portfolio_greeks["vega"]) > CONSTRAINTS["max_vega"]:
        warnings.append(f"⚠️ Vega limit: {portfolio_greeks['vega']:.1f} (max ±{CONSTRAINTS['max_vega']})")

    return warnings


def advance_round(state: GameState) -> dict:
    """Advance one round of market moves."""
    start_day = state.days_elapsed
    end_day = start_day + state.days_per_round
    round_returns = state.market_data[start_day:end_day]

    # Wealth before = cash + mark-to-market
    spot_before = state.spot_path[-1]
    wealth_before = state.capital + state.portfolio_value()

    # Advance spot through the week
    spot = spot_before
    for dS_pct in round_returns:
        spot += spot * dS_pct
    state.spot_path.append(spot)

    # Update implied vol
    if end_day < len(state.vol_data):
        state.implied_vol = state.vol_data[end_day - 1]
    else:
        state.implied_vol *= 1 + np.random.normal(0, 0.02)
        state.implied_vol = np.clip(state.implied_vol, 0.05, 0.80)

    state.round += 1

    # Settle expired options to cash
    remaining_positions = []
    for pos in state.positions:
        if pos.contract.days_remaining(state.days_elapsed) <= 0 and pos.contract.opt_type != "stock":
            if pos.contract.opt_type == "call":
                intrinsic = max(spot - pos.contract.strike, 0)
            else:
                intrinsic = max(pos.contract.strike - spot, 0)
            state.capital += intrinsic * pos.quantity
        else:
            remaining_positions.append(pos)
    state.positions = remaining_positions

    # Wealth after = cash + new mark-to-market
    wealth_after = state.capital + state.portfolio_value()
    round_pnl = wealth_after - wealth_before

    state.peak_capital = max(state.peak_capital, wealth_after)
    drawdown_pct = (1 - wealth_after / state.peak_capital) * 100

    if drawdown_pct > CONSTRAINTS["max_drawdown_pct"]:
        state.breached_limit = f"Drawdown limit breached: {drawdown_pct:.1f}% > {CONSTRAINTS['max_drawdown_pct']}%"

    state.pnl_history.append({
        "round": state.round,
        "pnl": round_pnl,
        "cumulative": wealth_after - CONSTRAINTS["starting_capital"],
        "spot": spot,
        "vol": state.implied_vol * 100,
        "drawdown_pct": drawdown_pct,
    })

    return {
        "round_pnl": round_pnl,
        "spot_move_pct": (spot / spot_before - 1) * 100,
        "new_spot": spot,
        "new_vol": state.implied_vol * 100,
        "drawdown_pct": drawdown_pct,
    }


def score_game(state: GameState) -> dict:
    """Final scoring."""
    total_wealth = state.capital + state.portfolio_value()
    total_pnl = total_wealth - CONSTRAINTS["starting_capital"]
    weekly_pnls = np.array([h["pnl"] for h in state.pnl_history]) if state.pnl_history else np.array([0.0])
    sharpe = float(weekly_pnls.mean() / weekly_pnls.std() * np.sqrt(52)) if weekly_pnls.std() > 0 else 0.0
    max_dd = max((h["drawdown_pct"] for h in state.pnl_history), default=0)
    total_costs = sum(t.get("cost", 0) for t in state.trade_history)

    if state.breached_limit:
        grade, summary = "F", f"Game over — {state.breached_limit}"
    elif sharpe >= 3.0 and total_pnl > 0:
        grade, summary = "A+", "Exceptional risk-adjusted returns with tight discipline."
    elif sharpe >= 2.0 and total_pnl > 0:
        grade, summary = "A", "Strong risk-adjusted returns, good discipline."
    elif sharpe >= 1.0 and total_pnl > 0:
        grade, summary = "B", "Positive returns with reasonable risk management."
    elif total_pnl > 0:
        grade, summary = "C", "Profitable but volatile — a real PM would get a talking-to."
    elif total_pnl > -CONSTRAINTS["starting_capital"] * 0.05:
        grade, summary = "D", "Small loss. Review sizing and timing."
    else:
        grade, summary = "F", "Significant loss. Study the feedback below."

    return {
        "total_pnl": total_pnl,
        "return_pct": total_pnl / CONSTRAINTS["starting_capital"] * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "total_costs": total_costs,
        "n_trades": len(state.trade_history),
        "grade": grade,
        "summary": summary,
    }


def generate_feedback(state: GameState) -> list[str]:
    """Post-game educational feedback."""
    feedback = []
    score = score_game(state)

    if not state.trade_history:
        return ["You didn't trade. Express views to learn."]

    if score["n_trades"] > state.total_rounds * 4:
        feedback.append(
            "🔄 **Over-trading**: Each trade costs spread. Real PMs are selective."
        )
    if any(abs(t.get("qty", 0)) > 20 for t in state.trade_history):
        feedback.append(
            "📏 **Large positions**: Position sizing is everything."
        )
    has_short_gamma = any(t.get("qty", 0) < 0 and "Stock" not in t.get("trade", "") for t in state.trade_history)
    if has_short_gamma and score["max_drawdown_pct"] > 10:
        feedback.append(
            "💥 **Short gamma pain**: You sold options and hit a large drawdown. "
            "Short gamma profits in calm markets but blows up fast."
        )
    if score["sharpe"] > 2:
        feedback.append("⭐ **Excellent Sharpe**: Well-balanced risk vs return.")
    elif 0 < score["sharpe"] < 0.5:
        feedback.append("📊 **Low Sharpe**: Returns were volatile. Hedge unwanted Greeks more.")

    greeks = state.portfolio_greeks()
    if abs(greeks["delta"]) > 20:
        feedback.append("📐 **Unhedged delta**: If trading vol, hedge delta regularly.")
    if score["total_costs"] > max(abs(score["total_pnl"]) * 0.3, 50):
        feedback.append("💸 **High costs**: Transaction costs ate into P&L significantly.")
    if not feedback:
        feedback.append("Solid execution. Try different market regimes (seeds).")

    return feedback
