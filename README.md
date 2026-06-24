# Derivatives Risk Intuition Dashboard

An interactive Streamlit dashboard for building intuition about quantitative derivatives trading. Select which risks you want exposure to, and the system constructs a portfolio, hedges the rest, and simulates forward to show P&L attribution — all using production-realistic models and live market data.

## Quick Start

```bash
cd derivatives-dashboard
source .venv/bin/activate
streamlit run src/dashboard/ui/app.py
```

## How It Works

### The Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. SELECT RISK         2. BUILD PORTFOLIO      3. HEDGE               │
│                                                                         │
│  User picks which       Optimizer finds          Hedger neutralizes     │
│  Greeks to be           instruments that         everything else        │
│  long/short/neutral     maximize desired         using liquid           │
│                         Greeks (ignores rest)    instruments             │
│                                                                         │
│  4. VISUALIZE           5. SIMULATE                                     │
│                                                                         │
│  See decomposition:     Step forward in time,                           │
│  what contributes       see hedge degrade,                              │
│  what risk              P&L by Greek                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tab: Portfolio

After you select target Greeks in the sidebar and click "Build Portfolio", this tab shows:
- **Positions table** — which instruments were chosen, in what quantities
- **Total Greek exposure** — the aggregate risk profile
- **Stacked bar chart** — each instrument's contribution to each Greek

The builder *deliberately* does not hedge unwanted Greeks. This separation lets you see the raw exposure before hedging.

### Tab: Hedge

Shows how the hedger neutralizes unwanted risk:
- **Before/After table** — each Greek before and after hedging, with % reduction
- **Hedge instruments** — what was added (e.g., short stock for delta, short options for vega)
- **Waterfall chart** — visual of how each hedge instrument progressively reduces each Greek
- **Partial hedge warnings** — when Greeks are mechanically coupled (e.g., gamma+vega in vanilla options), the hedger flags residual risk it couldn't eliminate without destroying your desired exposure

The hedger uses a soft-constraint (penalty-based) optimizer rather than hard equality constraints. This mirrors how real desks operate: they find the *best achievable* hedge rather than declaring "impossible" when perfect isolation isn't feasible with vanilla instruments.

### Tab: Simulation

Runs a forward simulation with realistic market dynamics:
- **Real or synthetic data** — toggle historical SPX daily returns (downloaded once, cached) or stochastic random moves
- **Stochastic realized vol** — daily vol itself varies, so gamma P&L has genuine uncertainty
- **Rebalancing comparison** — run with vs without periodic hedge rebalancing, see the Sharpe impact
- **Transaction costs** — adjustable rebalance cost (bps) shows the friction vs hedge quality tradeoff
- **Market driver plots** — spot path, cumulative IV shift, rolling realized vol, rate moves
- **Greek evolution subplots** — every Greek in its own panel, comparing rebalanced vs unrebalanced
- **P&L attribution** — stacked bar showing daily contribution from each Greek
- Simulations up to 252 days (1 trading year)

### Tab: Trading Game 🎮

A portfolio management game where you trade through a market period:
- **Dynamic options market** — 30+ contracts (calls/puts at 5 strikes × 3 expiries) with Greeks computed via Black-Scholes
- **Greeks update realistically** — as spot moves, time passes, and vol changes, your positions' delta/gamma/vega/theta evolve
- **Options expire** — short-dated positions settle at intrinsic, forcing you to roll or accept decay
- **Realistic constraints** — drawdown limits, delta/vega caps, position limits, bid-ask spreads proportional to gamma
- **Build your own structures** — construct straddles, butterflies, calendars from individual legs
- **Risk-adjusted scoring** — graded on Sharpe ratio with educational post-game feedback
- **Historical replay** — trade through real SPX market periods (or synthetic)

### Tab: Education

Four interactive sections:
1. **Greeks Intuition** — interactive plots of how Greeks change with spot/vol/time
2. **Portfolio Optimizer** — the math (LP formulation, L1 regularization, solver)
3. **Hedging Engine** — soft-constraint QP formulation, partial hedges, coupling explanation
4. **What Provides What** — heatmap + payoff diagrams of option structures

## Architecture

```
src/dashboard/
├── data/               Market data fetching & caching
│   ├── equity.py       yfinance option chains (live, cached 15min)
│   ├── historical.py   SPX daily returns (download once, cached forever)
│   ├── fx.py           FX spot (live) + vol surface (bundled, delta-space)
│   └── rates.py        Bundled yield curve + swaption vols
│
├── models/             Pricing engines
│   ├── equity.py       QuantLib: implied vol surface → Dupire local vol
│   ├── fx.py           Garman-Kohlhagen + SABR calibration (pysabr)
│   ├── rates.py        Yield curve bootstrap + Bachelier swaption pricing
│   └── greeks.py       Bump-and-reprice Greeks (surface-consistent, not flat-vol BS)
│
├── portfolio/          Construction & hedging
│   ├── builder.py      cvxpy optimizer: maximize desired Greeks only
│   └── hedger.py       cvxpy soft-constraint optimizer: best-effort neutralization
│
├── game/               Trading game
│   └── engine.py       BS pricing, dynamic market, position tracking, scoring
│
├── simulation/
│   └── engine.py       Forward time-stepping with P&L attribution (2nd order Taylor)
│
└── ui/                 Streamlit frontend
    ├── app.py          Main entry, tab layout, instrument generation, education
    ├── risk_selector.py  Sidebar: asset class + Greek direction selection
    ├── portfolio_view.py Positions table + contribution charts
    ├── hedge_view.py     Before/after + waterfall visualization
    ├── simulation_view.py  Interactive sim with rebalance comparison
    └── game_view.py      Trading game interface with live market
```

## Models & Realism

| Asset Class | Pricing Model | Data Source | Discrepancy vs Production |
|---|---|---|---|
| Equity options | Local vol (Dupire) via QuantLib | yfinance (live, 15min delay) | Production uses stochastic-local-vol (SLV) |
| FX options | Garman-Kohlhagen + SABR | Spot live, vol surface bundled | Production uses live feeds (Bloomberg FXGO) |
| Rate swaptions | Bachelier + SABR (β=0.5) | Bundled yield curve + vols | Production uses OIS discounting, live vol cube |

### Key Modeling Choices

- **Greeks are computed via bump-and-reprice** (finite difference on the vol surface), not BS closed-form. This captures smile effects in delta/gamma.
- **Swaption vols are normal (Bachelier)** — market standard for USD since negative rates era.
- **FX vols are in delta-space** (ATM DNS, 25Δ RR/BF) — the actual quoting convention used globally.
- **The builder and hedger are separate optimizations** — the builder expresses a view, the hedger cleans up residuals. This mirrors how trading desks actually operate.
- **The hedger uses soft constraints** — penalty-based minimization rather than hard equality constraints. This guarantees a solution even when Greeks are coupled (e.g., gamma+vega in vanilla options) and produces partial hedges with clear residual attribution.

## Dependencies

- `QuantLib` — local vol surfaces, yield curve bootstrapping
- `pysabr` — SABR model calibration
- `yfinance` — live equity/FX data
- `cvxpy` — portfolio/hedge optimization (convex programming)
- `plotly` — interactive visualizations
- `streamlit` — dashboard framework
- `scipy` — statistical functions (normal distribution, root finding)

No configuration needed. All vol surfaces and yield curves use bundled data.
