"""Hedge visualization — waterfall chart and before/after table."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.portfolio.builder import Portfolio, GREEKS


def render_hedge_view(
    portfolio: Portfolio,
    hedge: Portfolio,
    before_after: dict,
    target_greeks: dict[str, str],
):
    """Visualize hedging: waterfall of how each instrument reduces risk."""
    if not hedge.positions:
        st.info(
            "No hedge needed — residual Greeks are already near zero, "
            "or the desired Greek is too coupled to hedge without destroying the position."
        )
        _show_before_after_table(before_after, target_greeks)
        return

    # Check hedge quality and flag partial hedges
    partial = []
    for g in GREEKS:
        if target_greeks.get(g) == "neutral":
            before_val = abs(before_after["before"].get(g, 0))
            after_val = abs(before_after["after"].get(g, 0))
            if before_val > 0.01 and after_val > before_val * 0.3:
                partial.append(g)
    if partial:
        st.warning(
            f"⚠️ Partial hedge: {', '.join(g.capitalize() for g in partial)} could not be fully "
            f"neutralized without destroying your desired exposure. This is realistic — on a real "
            f"desk you'd accept residual risk or use exotic structures (variance swaps, corridors)."
        )

    st.subheader("🛡️ Hedge Overlay")
    _show_before_after_table(before_after, target_greeks)

    st.markdown("**Hedge Instruments**")
    hedge_rows = hedge.greeks_table()
    if hedge_rows:
        st.dataframe(pd.DataFrame(hedge_rows), use_container_width=True, hide_index=True)

    st.subheader("Hedge Waterfall")
    fig = _build_waterfall(before_after, target_greeks)
    st.plotly_chart(fig, use_container_width=True)


def _show_before_after_table(before_after: dict, target_greeks: dict):
    st.markdown("**Greek Profile: Before vs After Hedging**")
    rows = []
    for g in GREEKS:
        before = before_after["before"][g]
        after = before_after["after"][g]
        status = target_greeks.get(g, "neutral")
        rows.append({
            "Greek": g.capitalize(),
            "Target": status.upper(),
            "Before": f"{before:.4f}",
            "After": f"{after:.4f}",
            "Reduction": f"{(1 - abs(after)/(abs(before)+1e-9))*100:.0f}%",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_waterfall(before_after: dict, target_greeks: dict) -> go.Figure:
    hedged_greeks = [g for g in GREEKS if target_greeks.get(g) == "neutral"]
    if not hedged_greeks:
        hedged_greeks = GREEKS

    names = []
    values = []
    for g in hedged_greeks:
        before = before_after["before"][g]
        after = before_after["after"][g]
        names.append(f"{g.capitalize()} (before)")
        values.append(before)
        names.append(f"{g.capitalize()} hedge")
        values.append(after - before)

    fig = go.Figure(go.Waterfall(
        name="Hedge Effect",
        orientation="v",
        measure=["absolute" if i % 2 == 0 else "relative" for i in range(len(values))],
        x=names,
        y=values,
        connector={"line": {"color": "rgba(63,63,63,0.3)"}},
    ))
    fig.update_layout(height=400, title="Greek Reduction via Hedging")
    return fig
