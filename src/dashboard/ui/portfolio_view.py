"""Portfolio view — table and chart of portfolio Greeks."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.portfolio.builder import Portfolio, GREEKS


def render_portfolio_view(portfolio: Portfolio):
    """Display portfolio instruments and Greek breakdown."""
    if not portfolio.positions:
        st.warning("No portfolio constructed yet. Select risk exposures and click Build.")
        return

    st.subheader("📊 Portfolio Composition")

    # Positions table
    rows = portfolio.greeks_table()
    df = pd.DataFrame(rows)
    display_cols = ["instrument", "asset_class", "quantity"] + GREEKS
    df = df[[c for c in display_cols if c in df.columns]]

    # Format numbers
    for g in GREEKS:
        if g in df.columns:
            df[g] = df[g].apply(lambda x: f"{x:.4f}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    # Total Greeks
    totals = portfolio.total_greeks
    st.subheader("📈 Total Greek Exposure")
    cols = st.columns(len(GREEKS))
    for i, g in enumerate(GREEKS):
        with cols[i]:
            st.metric(g.capitalize(), f"{totals[g]:.4f}")

    # Stacked bar chart of Greek contributions
    st.subheader("Greek Contribution by Instrument")
    fig = go.Figure()
    instruments = [r["instrument"] for r in rows]
    for g in GREEKS:
        values = [r.get(g, 0) for r in rows]
        if any(abs(v) > 1e-6 for v in values):
            fig.add_trace(go.Bar(name=g.capitalize(), x=instruments, y=values))

    fig.update_layout(barmode="relative", height=400, xaxis_title="Instrument", yaxis_title="Greek Value")
    st.plotly_chart(fig, use_container_width=True)
