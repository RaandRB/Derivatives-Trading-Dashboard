"""Simulation view — dynamic hedge P&L attribution over time."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.portfolio.builder import Portfolio, GREEKS
from dashboard.simulation.engine import simulate_forward, SimulationResult


def render_simulation_view(portfolio: Portfolio, spot: float):
    """Interactive simulation: step forward and see P&L attribution."""
    st.subheader("⏱️ Dynamic Hedge Simulation")
    st.markdown(
        "Simulate the portfolio forward in time to see how the hedge degrades "
        "and where P&L comes from. Compare rebalancing every N days vs no rebalancing."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_days = st.slider("Days to simulate", 1, 252, 30)
    with col2:
        daily_vol = st.slider("Daily spot vol (%)", 0.5, 5.0, 1.5) / 100
    with col3:
        seed = st.number_input("Random seed", value=42, step=1)

    col4, col5, col6 = st.columns(3)
    with col4:
        rebalance_n = st.selectbox(
            "Rebalance every N days",
            [None, 1, 2, 3, 5, 7, 10],
            format_func=lambda x: "Never (compare)" if x is None else f"Every {x}d",
            index=3,
        )
    with col5:
        rebal_cost = st.slider(
            "Rebalance cost (bps)",
            0, 100, 10,
            help="Bid-ask + impact per rebalance. SPY stock ~1-2bps, options ~10-30bps.",
        )
    with col6:
        use_historical = st.checkbox(
            "Use real SPX data",
            help="Sample spot moves from historical S&P 500 daily returns (downloaded once via yfinance).",
        )

    if use_historical:
        st.caption("📊 Spot moves sampled from real SPX history. Daily vol slider is ignored.")

    if st.button("▶️ Run Simulation", type="primary"):
        # Always run no-rebalance as baseline
        result_no_rebal = simulate_forward(
            portfolio, spot, n_days=n_days, spot_vol=daily_vol, seed=int(seed),
            rebalance_every=None, use_historical=use_historical,
        )
        result_rebal = None
        if rebalance_n is not None:
            result_rebal = simulate_forward(
                portfolio, spot, n_days=n_days, spot_vol=daily_vol, seed=int(seed),
                rebalance_every=rebalance_n, rebalance_cost_bps=rebal_cost,
                use_historical=use_historical,
            )
        _display_results(result_no_rebal, result_rebal, rebalance_n)


def _display_results(
    result: SimulationResult,
    result_rebal: SimulationResult | None,
    rebalance_n: int | None,
):
    """Display simulation results with comparison charts."""
    if not result.steps:
        st.warning("No simulation steps.")
        return

    # Summary metrics
    final = result.steps[-1]
    daily_pnls = [s.pnl_total for s in result.steps]
    sharpe_no = _sharpe(daily_pnls)

    if result_rebal:
        final_r = result_rebal.steps[-1]
        daily_pnls_r = [s.pnl_total for s in result_rebal.steps]
        sharpe_re = _sharpe(daily_pnls_r)
        cols = st.columns(5)
        with cols[0]:
            st.metric("P&L (no rebalance)", f"${final.cumulative_pnl:,.2f}")
        with cols[1]:
            st.metric("Sharpe (no rebal)", f"{sharpe_no:.2f}")
        with cols[2]:
            st.metric(f"P&L (rebal {rebalance_n}d)", f"${final_r.cumulative_pnl:,.2f}")
        with cols[3]:
            st.metric(f"Sharpe (rebal {rebalance_n}d)", f"{sharpe_re:.2f}",
                      delta=f"{sharpe_re - sharpe_no:+.2f}")
        with cols[4]:
            st.metric("Rebalance benefit", f"${final_r.cumulative_pnl - final.cumulative_pnl:,.2f}")
    else:
        cols = st.columns(3)
        with cols[0]:
            st.metric("P&L (no rebalance)", f"${final.cumulative_pnl:,.2f}")
        with cols[1]:
            st.metric("Sharpe", f"{sharpe_no:.2f}")
        with cols[2]:
            st.metric("Max Drawdown", f"${min(s.cumulative_pnl for s in result.steps):,.2f}")

    # Cumulative P&L comparison
    st.markdown("**Cumulative P&L**")
    days = [s.day for s in result.steps]
    mode = "lines" if len(days) > 30 else "lines+markers"
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=days, y=[s.cumulative_pnl for s in result.steps],
        mode=mode, name="No rebalance", line=dict(color="#EF553B"),
    ))
    if result_rebal:
        fig_cum.add_trace(go.Scatter(
            x=days, y=[s.cumulative_pnl for s in result_rebal.steps],
            mode=mode, name=f"Rebalance every {rebalance_n}d", line=dict(color="#636EFA"),
        ))
    fig_cum.update_layout(height=300, xaxis_title="Day", yaxis_title="P&L ($)", legend=dict(x=0, y=1))
    st.plotly_chart(fig_cum, use_container_width=True)

    # P&L attribution stacked bar
    st.markdown("**Daily P&L Attribution by Greek**")
    fig_attr = go.Figure()
    source = result_rebal if result_rebal else result
    for g in GREEKS:
        values = [s.pnl_attribution.get(g, 0) for s in source.steps]
        if any(abs(v) > 0.01 for v in values):
            fig_attr.add_trace(go.Bar(name=g.capitalize(), x=days, y=values))
    fig_attr.add_trace(go.Bar(name="Residual", x=days, y=[s.residual for s in source.steps]))
    fig_attr.update_layout(barmode="relative", height=400, xaxis_title="Day", yaxis_title="P&L ($)")
    st.plotly_chart(fig_attr, use_container_width=True)

    # Greek evolution — all Greeks in subplots
    st.markdown("**Greek Evolution (hedge degradation)**")
    _plot_greek_evolution(result, result_rebal, rebalance_n)

    # Market drivers — spot, vol, rates that affect your P&L
    st.markdown("**Market Drivers (what moves your P&L)**")
    _plot_market_drivers(result)

    # Market moves table
    with st.expander("📋 Market Moves & P&L Detail"):
        detail = []
        for s in result.steps:
            detail.append({
                "Day": s.day,
                "Spot Δ%": f"{s.spot_change_pct:.2f}%",
                "Vol Δ%": f"{s.vol_change_pct:.3f}%",
                "Rate Δbp": f"{s.rate_change_bp:.1f}",
                "P&L": f"${s.pnl_total:,.2f}",
                "Cumulative": f"${s.cumulative_pnl:,.2f}",
            })
        st.dataframe(pd.DataFrame(detail), use_container_width=True, hide_index=True)


def _plot_greek_evolution(
    result: SimulationResult,
    result_rebal: SimulationResult | None,
    rebalance_n: int | None,
):
    """Subplot per Greek showing evolution over time, with rebalance comparison."""
    # Group into primary (large) and secondary (small) Greeks for layout
    primary = ["delta", "gamma", "vega", "theta"]
    secondary = [g for g in GREEKS if g not in primary
                 and any(abs(s.greeks_after.get(g, 0)) > 0.001 for s in result.steps)]
    all_greeks = primary + secondary
    n_plots = len(all_greeks)
    rows = (n_plots + 1) // 2

    fig = make_subplots(
        rows=rows, cols=2,
        subplot_titles=[g.capitalize() for g in all_greeks],
        vertical_spacing=0.08,
    )

    colors_no = "#EF553B"
    colors_re = "#636EFA"
    days = list(range(len(result.steps) + 1))
    mode = "lines" if len(days) > 30 else "lines+markers"

    for idx, g in enumerate(all_greeks):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # No-rebalance trace
        vals = [result.initial_greeks.get(g, 0)] + [s.greeks_after.get(g, 0) for s in result.steps]
        fig.add_trace(go.Scatter(
            x=days, y=vals, mode=mode, name="No rebal",
            line=dict(color=colors_no, width=2),
            legendgroup="no_rebal", showlegend=(idx == 0),
        ), row=row, col=col)

        # Rebalanced trace
        if result_rebal:
            vals_r = [result_rebal.initial_greeks.get(g, 0)] + [s.greeks_after.get(g, 0) for s in result_rebal.steps]
            fig.add_trace(go.Scatter(
                x=days, y=vals_r, mode=mode, name=f"Rebal {rebalance_n}d",
                line=dict(color=colors_re, width=2),
                legendgroup="rebal", showlegend=(idx == 0),
            ), row=row, col=col)

        # Zero reference line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.4, row=row, col=col)

    fig.update_layout(
        height=250 * rows,
        legend=dict(orientation="h", x=0.3, y=1.02),
        margin=dict(t=60),
    )
    fig.update_xaxes(title_text="Day", row=rows)
    st.plotly_chart(fig, use_container_width=True)


def _plot_market_drivers(result: SimulationResult):
    """Plot spot price, implied vol shift, and rate shift — the inputs driving P&L."""
    import numpy as np

    days = [s.day for s in result.steps]
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Spot Price", "Implied Vol Shift (%)", "Rate Shift (bp)"],
        vertical_spacing=0.08,
        shared_xaxes=True,
    )

    # Spot
    spot_levels = [s.spot_level for s in result.steps]
    fig.add_trace(go.Scatter(
        x=days, y=spot_levels, mode="lines", name="Spot",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=1)

    # Realized vol (rolling 5-day)
    vol_levels = [s.vol_level for s in result.steps]
    fig.add_trace(go.Scatter(
        x=days, y=vol_levels, mode="lines", name="Cum IV shift",
        line=dict(color="#AB63FA", width=2),
    ), row=2, col=1)
    # Also show rolling realized vol
    returns = np.array([s.spot_change_pct / 100 for s in result.steps])
    window = min(5, len(returns))
    if len(returns) >= window:
        realized = [np.std(returns[max(0, i-window+1):i+1]) * 100
                    for i in range(len(returns))]
        fig.add_trace(go.Scatter(
            x=days, y=realized, mode="lines", name=f"Realized vol ({window}d, %)",
            line=dict(color="#FFA15A", width=1.5, dash="dash"),
        ), row=2, col=1)

    # Rates
    rate_levels = [s.rate_level_bp for s in result.steps]
    fig.add_trace(go.Scatter(
        x=days, y=rate_levels, mode="lines", name="Cum rate shift",
        line=dict(color="#00CC96", width=2),
    ), row=3, col=1)

    fig.update_layout(height=500, showlegend=True, legend=dict(orientation="h", y=1.02))
    fig.update_xaxes(title_text="Day", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)


def _sharpe(daily_pnls: list[float]) -> float:
    """Annualized Sharpe ratio from daily P&L series."""
    import numpy as np
    arr = np.array(daily_pnls)
    if arr.std() == 0:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))
