"""Derivatives Risk Intuition Dashboard — Main Entry Point.

Run with: streamlit run src/dashboard/ui/app.py
"""

import streamlit as st
import numpy as np
from datetime import date

st.set_page_config(page_title="Derivatives Risk Dashboard", layout="wide", page_icon="📈")

from dashboard.ui.risk_selector import render_risk_selector
from dashboard.ui.portfolio_view import render_portfolio_view
from dashboard.ui.hedge_view import render_hedge_view
from dashboard.ui.simulation_view import render_simulation_view
from dashboard.ui.game_view import render_game_view
from dashboard.portfolio.builder import Portfolio, Instrument, build_portfolio, GREEKS
from dashboard.portfolio.hedger import hedge_portfolio


def main():
    st.title("📈 Derivatives Risk Intuition Dashboard")
    st.markdown(
        "Select which risk factors you want exposure to. The system builds a portfolio "
        "from real instruments, hedges the rest, and simulates forward to show P&L attribution."
    )

    # Sidebar: risk selection
    config = render_risk_selector()
    target_greeks = config["target_greeks"]
    ticker = config["ticker"]

    # Main area tabs
    tab_portfolio, tab_hedge, tab_sim, tab_game, tab_edu = st.tabs([
        "Portfolio", "Hedge", "Simulation", "🎮 Trading Game", "Education"
    ])

    # Generate instruments (using representative Greeks when live data unavailable)
    instruments = _generate_instruments(config)
    spot = 550.0  # fallback; overwritten by live data when available

    # Try to fetch live spot
    try:
        from dashboard.data.equity import get_spot
        spot = get_spot(ticker)
    except Exception:
        st.sidebar.warning(f"Could not fetch live spot for {ticker}, using ${spot:.0f}")

    # Build portfolio on button click
    if st.sidebar.button("🔨 Build Portfolio", type="primary"):
        portfolio = build_portfolio(instruments, target_greeks)
        st.session_state["portfolio"] = portfolio
        st.session_state["spot"] = spot

        # Auto-hedge neutral Greeks
        greeks_to_hedge = [g for g, d in target_greeks.items() if d == "neutral"]
        if greeks_to_hedge:
            hedge_insts = _generate_hedge_instruments(config, spot)
            hedge, before_after = hedge_portfolio(portfolio, hedge_insts, greeks_to_hedge)
            st.session_state["hedge"] = hedge
            st.session_state["before_after"] = before_after
        else:
            st.session_state["hedge"] = Portfolio()
            st.session_state["before_after"] = {"before": portfolio.total_greeks, "after": portfolio.total_greeks}

    # Render views
    portfolio = st.session_state.get("portfolio", Portfolio())
    hedge = st.session_state.get("hedge", Portfolio())
    before_after = st.session_state.get("before_after", {})
    spot = st.session_state.get("spot", spot)

    with tab_portfolio:
        render_portfolio_view(portfolio)

    with tab_hedge:
        if before_after:
            render_hedge_view(portfolio, hedge, before_after, target_greeks)
        else:
            st.info("Build a portfolio first to see the hedge.")

    with tab_sim:
        # Combined portfolio + hedge for simulation
        combined = Portfolio(positions=portfolio.positions + hedge.positions)
        render_simulation_view(combined, spot)

    with tab_game:
        render_game_view()

    with tab_edu:
        _render_education()


def _generate_instruments(config: dict) -> list[Instrument]:
    """Generate available instruments with representative Greeks.

    In a full implementation, these would be computed from live option chains
    using the local vol / SABR models. Here we use realistic representative
    values to demonstrate the optimization.
    """
    instruments = []
    spot = 550.0  # SPY approximate

    if config["asset_classes"].get("equity"):
        # ATM call
        instruments.append(Instrument(
            name=f"{config['ticker']} ATM Call", asset_class="equity", instrument_type="call",
            greeks={"delta": 0.52, "gamma": 0.008, "vega": 0.35, "theta": -0.15, "rho": 0.12, "vanna": 0.01, "volga": 0.02},
        ))
        # ATM put
        instruments.append(Instrument(
            name=f"{config['ticker']} ATM Put", asset_class="equity", instrument_type="put",
            greeks={"delta": -0.48, "gamma": 0.008, "vega": 0.35, "theta": -0.14, "rho": -0.10, "vanna": -0.01, "volga": 0.02},
        ))
        # OTM call (higher strike)
        instruments.append(Instrument(
            name=f"{config['ticker']} 105% Call", asset_class="equity", instrument_type="call",
            greeks={"delta": 0.30, "gamma": 0.012, "vega": 0.28, "theta": -0.10, "rho": 0.07, "vanna": 0.03, "volga": 0.05},
        ))
        # OTM put (lower strike)
        instruments.append(Instrument(
            name=f"{config['ticker']} 95% Put", asset_class="equity", instrument_type="put",
            greeks={"delta": -0.30, "gamma": 0.012, "vega": 0.28, "theta": -0.10, "rho": -0.07, "vanna": -0.03, "volga": 0.05},
        ))
        # Underlying (for delta hedging)
        instruments.append(Instrument(
            name=f"{config['ticker']} Stock", asset_class="equity", instrument_type="underlying",
            greeks={"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0},
        ))
        # Straddle (long vol)
        instruments.append(Instrument(
            name=f"{config['ticker']} ATM Straddle", asset_class="equity", instrument_type="call",
            greeks={"delta": 0.04, "gamma": 0.016, "vega": 0.70, "theta": -0.29, "rho": 0.02, "vanna": 0.0, "volga": 0.04},
        ))

    if config["asset_classes"].get("fx"):
        pair = config["fx_pair"]
        instruments.append(Instrument(
            name=f"{pair} 1M ATM Call", asset_class="fx", instrument_type="call",
            greeks={"delta": 0.50, "gamma": 15.0, "vega": 0.0020, "theta": -0.0003, "rho": 0.005, "vanna": 0.5, "volga": 0.1},
        ))
        instruments.append(Instrument(
            name=f"{pair} 1M ATM Put", asset_class="fx", instrument_type="put",
            greeks={"delta": -0.50, "gamma": 15.0, "vega": 0.0020, "theta": -0.0003, "rho": -0.005, "vanna": -0.5, "volga": 0.1},
        ))
        instruments.append(Instrument(
            name=f"{pair} 1M 25D RR", asset_class="fx", instrument_type="call",
            greeks={"delta": 0.25, "gamma": 5.0, "vega": 0.0008, "theta": -0.0001, "rho": 0.002, "vanna": 1.2, "volga": 0.3},
        ))

    if config["asset_classes"].get("rates"):
        instruments.append(Instrument(
            name="5Y10Y Payer Swaption", asset_class="rates", instrument_type="swaption",
            greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.45, "theta": -0.05, "rho": 0.85, "vanna": 0.0, "volga": 0.03},
        ))
        instruments.append(Instrument(
            name="2Y5Y Receiver Swaption", asset_class="rates", instrument_type="swaption",
            greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.30, "theta": -0.03, "rho": -0.55, "vanna": 0.0, "volga": 0.02},
        ))
        instruments.append(Instrument(
            name="10Y IRS (pay fixed)", asset_class="rates", instrument_type="swap",
            greeks={"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.90, "vanna": 0.0, "volga": 0.0},
        ))

    return instruments


def _generate_hedge_instruments(config: dict, spot: float) -> list[Instrument]:
    """Liquid instruments available for hedging — includes OTM options for volga/vanna."""
    hedges = [
        Instrument(
            name=f"{config['ticker']} Stock", asset_class="equity", instrument_type="underlying",
            greeks={"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0},
        ),
        Instrument(
            name=f"{config['ticker']} ATM Call (hedge)", asset_class="equity", instrument_type="call",
            greeks={"delta": 0.52, "gamma": 0.008, "vega": 0.35, "theta": -0.15, "rho": 0.12, "vanna": 0.01, "volga": 0.02},
        ),
        Instrument(
            name=f"{config['ticker']} ATM Put (hedge)", asset_class="equity", instrument_type="put",
            greeks={"delta": -0.48, "gamma": 0.008, "vega": 0.35, "theta": -0.14, "rho": -0.10, "vanna": -0.01, "volga": 0.02},
        ),
        # OTM options — higher vanna/volga, useful for hedging higher-order Greeks
        Instrument(
            name=f"{config['ticker']} 110% Call (hedge)", asset_class="equity", instrument_type="call",
            greeks={"delta": 0.20, "gamma": 0.010, "vega": 0.22, "theta": -0.07, "rho": 0.04, "vanna": 0.05, "volga": 0.08},
        ),
        Instrument(
            name=f"{config['ticker']} 90% Put (hedge)", asset_class="equity", instrument_type="put",
            greeks={"delta": -0.20, "gamma": 0.010, "vega": 0.22, "theta": -0.07, "rho": -0.04, "vanna": -0.05, "volga": 0.08},
        ),
    ]
    if config["asset_classes"].get("fx"):
        hedges.append(Instrument(
            name=f"{config['fx_pair']} Spot", asset_class="fx", instrument_type="underlying",
            greeks={"delta": 1.0, "gamma": 0, "vega": 0, "theta": 0, "rho": 0, "vanna": 0, "volga": 0},
        ))
    return hedges


def _render_education():
    """Educational content: math behind the optimizer, hedging, and option intuition."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    section = st.radio(
        "Topic", ["Option Greeks Intuition", "Portfolio Optimizer", "Hedging Engine", "What Provides What"],
        horizontal=True,
    )

    if section == "Option Greeks Intuition":
        _edu_greeks_intuition()
    elif section == "Portfolio Optimizer":
        _edu_optimizer()
    elif section == "Hedging Engine":
        _edu_hedging()
    elif section == "What Provides What":
        _edu_what_provides_what()


def _edu_greeks_intuition():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.stats import norm

    st.subheader("📚 Understanding the Greeks")
    st.markdown("""
**Delta (Δ)** — How much the option value changes per $1 move in the underlying.
A delta of 0.5 means the option gains \\$0.50 when the stock rises \\$1.

**Gamma (Γ)** — Rate of change of delta. High gamma means your delta changes rapidly
with spot moves. This is why gamma-scalping works: you rebalance delta frequently
and profit from the convexity.

**Vega (ν)** — Sensitivity to implied volatility. Long vega profits when vol rises.
In practice, vega varies by strike (the "smile") — this isn't captured by flat-vol BS.

**Theta (Θ)** — Daily time decay. Long options bleed theta; short options collect it.
Theta is the "cost" of holding gamma/vega exposure. There's no free lunch: you pay
theta to own convexity.

**Rho (ρ)** — Sensitivity to interest rates. More significant for longer-dated options
and dominant for rates instruments (swaps, swaptions).

**Vanna** — Cross-derivative: ∂Δ/∂σ = ∂ν/∂S. When vol rises, delta shifts.
This is the skew effect. Zero in flat-vol BS, non-zero in reality.

**Volga (Vomma)** — ∂ν/∂σ = ∂²V/∂σ². OTM options have positive volga — they
benefit disproportionately from vol spikes.
    """)

    # Interactive: show how Greeks change with spot
    st.subheader("Interactive: Greeks vs Spot Price")
    col1, col2, col3 = st.columns(3)
    with col1:
        K = st.slider("Strike", 80.0, 120.0, 100.0, key="edu_K")
    with col2:
        vol = st.slider("Volatility (%)", 5, 80, 20, key="edu_vol") / 100
    with col3:
        T = st.slider("Days to Expiry", 5, 365, 60, key="edu_T") / 365

    spots = np.linspace(60, 140, 200)
    r = 0.04

    deltas, gammas, vegas, thetas = [], [], [], []
    for S in spots:
        d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
        d2 = d1 - vol * np.sqrt(T)
        deltas.append(norm.cdf(d1))
        gammas.append(norm.pdf(d1) / (S * vol * np.sqrt(T)))
        vegas.append(S * norm.pdf(d1) * np.sqrt(T) / 100)
        thetas.append(-(S * norm.pdf(d1) * vol) / (2 * np.sqrt(T)) / 365)

    fig = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])
    fig.add_trace(go.Scatter(x=spots, y=deltas, name="Delta"), row=1, col=1)
    fig.add_trace(go.Scatter(x=spots, y=gammas, name="Gamma"), row=1, col=2)
    fig.add_trace(go.Scatter(x=spots, y=vegas, name="Vega"), row=2, col=1)
    fig.add_trace(go.Scatter(x=spots, y=thetas, name="Theta"), row=2, col=2)

    # Mark the strike
    for row, col in [(1,1),(1,2),(2,1),(2,2)]:
        fig.add_vline(x=K, line_dash="dash", line_color="gray", row=row, col=col)

    fig.update_layout(height=500, showlegend=False, title_text="Call Option Greeks vs Spot (dashed = strike)")
    st.plotly_chart(fig, use_container_width=True)


def _edu_optimizer():
    st.subheader("🧮 Portfolio Construction Optimizer")
    st.markdown(r"""
### The Problem

You want exposure to specific Greeks (e.g., long Gamma, long Vega) but you have
a menu of instruments that each carry a *bundle* of Greeks. Buying a call gives you
delta AND gamma AND vega AND theta. You can't buy just one Greek in isolation.

### Mathematical Formulation

Given $n$ available instruments with Greek matrix $G \in \mathbb{R}^{n \times 7}$
(each row is one instrument's Greeks), we solve for position quantities $w \in \mathbb{R}^n$:

$$\max_w \sum_{g \in \text{desired}} \text{sign}(g) \cdot (G_g^T w) - \lambda \|w\|_1$$

**subject to:**
- For "long" Greeks: $G_g^T w \geq \epsilon$ (ensure positive exposure)
- For "short" Greeks: $G_g^T w \leq -\epsilon$ (ensure negative exposure)
- Position limits: $\|w\|_\infty \leq W_{\max}$

### What This Means in English

1. **Maximize desired Greeks** — if you want long gamma, the optimizer finds
   combinations that produce the most gamma per unit of cost (regularization).

2. **No neutrality constraint** — unlike the hedger, the builder does NOT try to
   zero out unwanted Greeks. It just maximizes what you want.

3. **L1 regularization** ($\lambda \|w\|_1$) — penalizes large positions,
   preferring simpler portfolios with fewer instruments. This is the same idea
   as LASSO regression.

4. **Position limits** — prevents degenerate solutions with infinite leverage.

### Why Not Just Buy the Instrument with the Most Gamma?

Because you might want gamma AND vega simultaneously. A straddle gives both,
but an OTM put gives gamma with less vega. The optimizer finds the best
linear combination across all available instruments.

### Solver

We use CVXPY with the ECOS solver (interior-point method for conic programs).
The problem is convex because we maximize a linear objective with linear
constraints and a convex penalty term.
    """)

    # Visual example
    st.subheader("Example: Building a Long-Gamma Portfolio")
    st.markdown("""
    | Instrument | Delta | Gamma | Vega | Theta |
    |---|---|---|---|---|
    | ATM Call | 0.52 | 0.008 | 0.35 | -0.15 |
    | ATM Put | -0.48 | 0.008 | 0.35 | -0.14 |
    | Stock | 1.0 | 0 | 0 | 0 |
    | Straddle | 0.04 | 0.016 | 0.70 | -0.29 |

    **Target: Long Gamma.** The optimizer picks the straddle (highest gamma/dollar)
    and may also buy ATM calls/puts. Notice that all of these also carry vega and
    theta — the hedger's job is to deal with those residuals later.
    """)


def _edu_hedging():
    st.subheader("🛡️ Hedging Engine")
    st.markdown(r"""
### The Problem

After portfolio construction, you have unwanted Greek residuals
(e.g., delta = 76, theta = -7.5). The hedger neutralizes these using
liquid instruments while:
1. Minimizing transaction costs
2. Not disturbing the Greeks you *want* to keep

### Mathematical Formulation

Given portfolio Greeks $g_0 \in \mathbb{R}^7$ and hedge instrument matrix
$H \in \mathbb{R}^{m \times 7}$, solve for hedge quantities $h \in \mathbb{R}^m$:

$$\min_h \quad \lambda_1 \sum_{g \in \text{hedged}} \frac{|g_{0,g} + H_g^T h|}{s_g}
\;+\; \lambda_2 \sum_{g \notin \text{hedged}} \frac{|H_g^T h|}{s_g}
\;+\; \lambda_3 \|h\|_1$$

Where $s_g = \max(|g_{0,g}|, 0.5)$ normalizes each Greek, and
$\lambda_1 = 10$, $\lambda_2 = 1$, $\lambda_3 = 0.01$ encode priority.

### Why Soft Constraints (Not Hard)

The previous version used hard constraints ($|residual| \leq \tau$), which
fails when Greeks are mechanically coupled. Example:

> **Long gamma, hedge everything else.** Gamma, vega, and theta are coupled
> in vanilla options — you cannot sell vega without also selling gamma.
> Hard constraints make this infeasible → "no hedge found."

A real desk never says "impossible." They find the *best achievable* hedge
and accept residual risk where Greeks are coupled. The soft-constraint
formulation always produces a solution ranked by priority:

1. **Minimize residual on hedged Greeks** ($\lambda_1 = 10$, highest weight) —
   the Greeks you don't want get driven as close to zero as possible.

2. **Preserve desired Greeks** ($\lambda_2 = 1$) — the Greeks you *want* to keep
   are protected from disturbance. The optimizer won't destroy your gamma
   to perfectly zero your vega.

3. **Minimize cost** ($\lambda_3 = 0.01$) — prefer smaller positions as a
   tiebreaker (transaction costs / market impact).

### Partial Hedges and Coupled Greeks

When the optimizer can't fully neutralize a Greek without destroying your
desired exposure, it produces a **partial hedge**. The dashboard flags this:

| Scenario | What Gets Partially Hedged | Why |
|---|---|---|
| Long gamma, hedge vega | Vega ~95% hedged, gamma reduced ~30% | Options bundle gamma+vega |
| Long vega, hedge theta | Theta ~80% hedged | Long vol = long theta cost |
| Long vanna, hedge delta | Delta ~100%, vanna partially eaten | OTM delta shifts with vol |

**What a real desk would do for perfect isolation:**
- **Variance swaps** — pure vega, no gamma at inception
- **Gamma swaps** — pure realized vol exposure
- **VIX futures** — vol exposure decoupled from spot Greeks
- **Dynamic delta hedging** — rebalance frequently instead of statically

### The Hedging Hierarchy (What Hedges What)

| Greek to Hedge | Primary Instrument | Why |
|---|---|---|
| **Delta** | Stock / futures | Pure delta, no other Greeks. Cheapest hedge. |
| **Gamma** | ATM options | Highest gamma per dollar of premium |
| **Vega** | Longer-dated options | More vega per unit of theta cost |
| **Theta** | Short options | Selling options collects theta |
| **Vanna** | OTM options / risk reversals | Asymmetric delta-vol exposure |
| **Volga** | OTM wings (strangles) | Deep OTM options have highest volga |

### Why Sequential Hedging is Common in Practice

In reality, desks often hedge in priority order:
1. Delta-hedge first (cheap, stock/futures)
2. Gamma/Vega hedge with options
3. Higher-order adjustments if material

Our optimizer does it all at once (joint optimization), which is theoretically
optimal but can produce complex solutions. Production systems often split it
into sequential steps for operational simplicity.

### Solver Behavior

The formulation is a convex program (minimizing weighted sum of absolute values =
L1 penalties). Solved via ECOS (interior-point) with SCS as fallback. Because there
are no hard equality constraints, the problem is always feasible — the worst case
is "do nothing" ($h = 0$), which the optimizer will beat whenever any hedge helps.
    """)


def _edu_what_provides_what():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("🎯 What Option Structures Provide What Risk")
    st.markdown("""
    Each options structure gives you a different combination of Greeks.
    Understanding this mapping is the key insight for derivatives trading.
    """)

    # Visual: heatmap of structures × Greeks
    structures = [
        "Long Call", "Long Put", "Long Stock",
        "Straddle (long)", "Strangle (long)",
        "Call Spread (bull)", "Risk Reversal",
        "Butterfly (short wings)",
    ]
    greeks_list = ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Volga"]

    # Qualitative Greek profiles: -2 to +2 scale
    profiles = [
        [+2, +1, +1, -1, +0.5, +0.5],   # Long Call
        [-2, +1, +1, -1, -0.5, +0.5],    # Long Put
        [+2, 0, 0, 0, 0, 0],             # Long Stock
        [0, +2, +2, -2, 0, +1],          # Straddle
        [0, +1, +1.5, -1.5, 0, +2],     # Strangle
        [+1, 0, 0, -0.5, +0.5, -0.5],   # Call Spread
        [+1, 0, 0, 0, +2, -1],          # Risk Reversal
        [0, -2, -1, +1, 0, -2],         # Butterfly
    ]

    fig = go.Figure(data=go.Heatmap(
        z=profiles,
        x=greeks_list,
        y=structures,
        colorscale=[[0, "rgb(255,80,80)"], [0.5, "rgb(255,255,255)"], [1, "rgb(80,180,80)"]],
        zmid=0,
        text=[[f"{v:+.1f}" for v in row] for row in profiles],
        texttemplate="%{text}",
        textfont={"size": 14},
        showscale=False,
    ))
    fig.update_layout(
        height=400, title="Greek Profile by Structure (green=long, red=short)",
        xaxis_title="Greek", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed walkthrough
    st.markdown("""
---

### Structure-by-Structure Walkthrough

**Long Call** — The workhorse. Gives you delta (directional bet) + gamma (convexity) +
vega (vol exposure), but costs theta every day. A 0.5-delta ATM call is the most
balanced exposure to all three.

**Long Straddle (ATM Call + ATM Put)** — Delta-neutral by construction (≈0).
Pure gamma + vega play. You profit if the stock moves *either direction* more than
implied vol suggests, or if realized vol increases. Costs significant theta.
*This is the classic "long vol" trade.*

**Long Strangle (OTM Call + OTM Put)** — Similar to straddle but cheaper (OTM
options cost less). Lower gamma but HIGHER volga — you benefit more from vol
spikes because OTM options are more sensitive to vol-of-vol.
*Preferred when you think there might be a vol explosion (tail event).*

**Bull Call Spread (Long lower-strike call, Short higher-strike call)** — Capped
upside. Almost zero gamma and vega (they offset). Mostly a delta/directional trade
with limited premium at risk. The short wing finances the long wing.
*Use when you have a directional view but vol is expensive.*

**Risk Reversal (Long OTM Call, Short OTM Put)** — Strong delta + vanna exposure.
When vol rises, your delta becomes more positive (vanna effect from the skew).
Zero premium (approximately), so it's a leveraged directional bet that benefits
from vol increasing.
*Common in FX markets to express a directional view cheaply.*

**Short Butterfly (Short wings, Long body)** — Negative gamma (you lose from large
moves), positive theta (you collect time decay). Negative volga (you lose from
vol spikes). *The income-generation trade — you're selling insurance.*

---

### The Fundamental Tradeoff

| You Want | You Pay |
|---|---|
| Gamma (convexity) | Theta (time decay) |
| Vega (vol exposure) | Theta (time decay) |
| Volga (tail protection) | Vega + Theta |
| Delta (direction) | Nothing if via stock; theta if via options |

**There is no free lunch.** Every Greek exposure has a cost. The dashboard
helps you see exactly what you're paying for what you're getting.
    """)

    # Payoff diagrams
    st.subheader("Payoff Diagrams at Expiry")
    spots = np.linspace(80, 120, 200)
    K = 100

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Long Call (K=100)", "Long Straddle (K=100)",
        "Bull Call Spread (95/105)", "Risk Reversal (95P/105C)",
    ])

    # Long call
    fig.add_trace(go.Scatter(x=spots, y=np.maximum(spots - K, 0) - 5, name="Long Call"), row=1, col=1)
    # Straddle
    fig.add_trace(go.Scatter(x=spots, y=np.maximum(spots - K, 0) + np.maximum(K - spots, 0) - 10, name="Straddle"), row=1, col=2)
    # Bull call spread
    fig.add_trace(go.Scatter(x=spots, y=np.maximum(spots - 95, 0) - np.maximum(spots - 105, 0) - 4, name="Call Spread"), row=2, col=1)
    # Risk reversal
    fig.add_trace(go.Scatter(x=spots, y=np.maximum(spots - 105, 0) - np.maximum(95 - spots, 0), name="Risk Rev"), row=2, col=2)

    for row, col in [(1,1),(1,2),(2,1),(2,2)]:
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
