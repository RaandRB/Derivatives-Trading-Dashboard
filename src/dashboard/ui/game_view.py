"""Trading Game UI — Streamlit interface for the portfolio management game."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.game.engine import (
    CONSTRAINTS, GREEKS,
    GameState, init_game, get_round_context, get_market, execute_trade,
    advance_round, score_game, generate_feedback,
)


def render_game_view():
    """Main game tab."""
    st.subheader("🎮 Trading Game")
    st.markdown(
        "Manage a derivatives book through a market period. Each round = 1 week. "
        "Trade real options with dynamic Greeks. Breach drawdown limits and you're fired."
    )

    if "game_state" not in st.session_state or st.session_state.get("game_over"):
        _render_setup()
    else:
        state = st.session_state["game_state"]
        if state.round >= state.total_rounds or state.breached_limit:
            _render_game_over(state)
        else:
            _render_round(state)


def _render_setup():
    """Game configuration screen."""
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Constraints (real PM rules):**
        - Starting capital: **${CONSTRAINTS['starting_capital']:,.0f}**
        - Max drawdown: **{CONSTRAINTS['max_drawdown_pct']}%** (breach = game over)
        - Max |delta|: **{CONSTRAINTS['max_delta']}**
        - Max |vega|: **{CONSTRAINTS['max_vega']}**
        - Max position: **±{CONSTRAINTS['max_position_units']} contracts**
        """)
    with col2:
        st.markdown("""
        **Scoring:**
        - **A+**: Sharpe ≥ 3, profitable
        - **A**: Sharpe ≥ 2, profitable
        - **B**: Sharpe ≥ 1, profitable
        - **C**: Profitable but volatile
        - **D/F**: Loss or limit breach
        """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_rounds = st.selectbox("Weeks to play", [10, 15, 20, 30, 40, 52], index=2)
    with col2:
        use_hist = st.checkbox("Use real SPX data", value=True)
    with col3:
        seed = st.number_input("Market seed", value=42, step=1)

    if st.button("🎲 Start Game", type="primary"):
        state = init_game(total_rounds=n_rounds, use_historical=use_hist, seed=int(seed))
        st.session_state["game_state"] = state
        st.session_state["game_over"] = False
        st.rerun()


def _render_round(state: GameState):
    """Active round — show context, market, positions, trade interface."""
    context = get_round_context(state)

    # Header metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Week", f"{context['round']} / {context['total_rounds']}")
    with col2:
        st.metric("Spot", f"{context['spot']:.2f}")
    with col3:
        st.metric("IV", f"{context['implied_vol']:.1f}%")
    with col4:
        st.metric("P&L", f"${context['pnl_so_far']:+,.0f}")
    with col5:
        dd = context["drawdown_pct"]
        st.metric("Drawdown", f"{dd:.1f}%")

    # Market context
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"5d return: {context['recent_5d_return']:+.2f}%")
    with col2:
        st.caption(f"Realized vol: {context.get('recent_20d_vol', context['recent_5d_vol']):.1f}%")
    with col3:
        st.caption(f"Trend: {context.get('trend', 'unknown')}")

    # Current positions
    st.markdown("---")
    st.markdown("**📖 Current Book**")
    positions = context["positions"]
    if positions:
        pos_df = pd.DataFrame(positions)
        pos_df = pos_df[["label", "qty", "days_left", "delta", "gamma", "vega", "theta", "pnl"]]
        pos_df.columns = ["Contract", "Qty", "Days Left", "Δ", "Γ", "ν", "Θ", "P&L"]
        for col in ["Δ", "Γ", "ν", "Θ", "P&L"]:
            pos_df[col] = pos_df[col].apply(lambda x: f"{x:.3f}")
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

        # Total Greeks
        greeks = context["current_greeks"]
        greek_str = "  |  ".join(f"**{g[:3].capitalize()}**: {greeks[g]:.2f}" for g in ["delta", "gamma", "vega", "theta"])
        st.markdown(greek_str)
    else:
        st.info("No positions. Use the market below to trade.")

    # Options Market
    st.markdown("---")
    st.markdown("**🏪 Options Market**")
    st.caption("+qty = buy (long), −qty = sell (short). Greeks shown are per contract long.")

    market = get_market(state)
    _render_market(state, market)

    # Action buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⏭️ Done Trading — Advance Week", type="primary"):
            result = advance_round(state)
            st.session_state["last_round_result"] = result
            st.rerun()
    with col2:
        pass
    with col3:
        if st.button("🏳️ End Game Early"):
            st.session_state["game_over"] = True
            st.rerun()

    # Last round result
    if "last_round_result" in st.session_state:
        r = st.session_state["last_round_result"]
        st.info(f"Last week: Spot {r['spot_move_pct']:+.2f}%, IV→{r['new_vol']:.1f}%, P&L ${r['round_pnl']:+,.2f}")

    # Progress chart
    if state.pnl_history:
        _plot_game_progress(state)


def _render_market(state: GameState, market: list[dict]):
    """Render the options market as a tradeable table."""
    # Group by expiry
    stock_item = market[0]  # stock is always first
    option_items = market[1:]

    # Stock trade
    with st.expander("📈 Stock (pure delta)", expanded=False):
        qty = st.number_input("Stock qty (+buy, −sell)", min_value=-50, max_value=50, value=0, step=1, key="trade_stock")
        if qty != 0 and st.button("Trade Stock", key="btn_stock"):
            warnings = execute_trade(state, stock_item["contract"], qty)
            for w in warnings:
                st.warning(w)
            st.rerun()

    # Options by expiry
    expiries = {}
    for item in option_items:
        exp_label = item["label"].split()[-1]  # "1W", "1M", "3M"
        expiries.setdefault(exp_label, []).append(item)

    for exp_label, items in expiries.items():
        with st.expander(f"📋 {exp_label} Options ({len(items)} contracts)", expanded=(exp_label == "1M")):
            # Show as table
            rows = []
            for item in items:
                g = item["greeks"]
                rows.append({
                    "Contract": item["label"],
                    "Price": f"{g['price']:.3f}",
                    "Spread": f"{item['bid_ask_spread']:.3f}",
                    "Δ": f"{g['delta']:.3f}",
                    "Γ": f"{g['gamma']:.4f}",
                    "ν": f"{g['vega']:.4f}",
                    "Θ": f"{g['theta']:.4f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Trade selector
            labels = [item["label"] for item in items]
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                selected = st.selectbox("Select", labels, key=f"sel_{exp_label}")
            with col2:
                qty = st.number_input("Qty", min_value=-50, max_value=50, value=0, step=1, key=f"qty_{exp_label}")
            with col3:
                if st.button("Trade", key=f"btn_{exp_label}", type="primary"):
                    if qty != 0:
                        item = next(i for i in items if i["label"] == selected)
                        warnings = execute_trade(state, item["contract"], qty)
                        for w in warnings:
                            st.warning(w)
                        st.rerun()


def _render_game_over(state: GameState):
    """End screen."""
    score = score_game(state)
    feedback = generate_feedback(state)

    grade_icons = {"A+": "🏆", "A": "🥇", "B": "🥈", "C": "🥉", "D": "😐", "F": "💀"}
    st.markdown(f"## {grade_icons.get(score['grade'], '')} Final Grade: {score['grade']}")
    st.markdown(f"*{score['summary']}*")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total P&L", f"${score['total_pnl']:+,.2f}")
    with col2:
        st.metric("Return", f"{score['return_pct']:+.1f}%")
    with col3:
        st.metric("Sharpe (ann.)", f"{score['sharpe']:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{score['max_drawdown_pct']:.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Trades", score["n_trades"])
    with col2:
        st.metric("Transaction Costs", f"${score['total_costs']:,.2f}")

    st.markdown("---")
    st.markdown("### 📚 Feedback")
    for f in feedback:
        st.markdown(f)

    st.markdown("---")
    _plot_game_progress(state)

    with st.expander("📋 Trade Log"):
        if state.trade_history:
            st.dataframe(pd.DataFrame(state.trade_history), use_container_width=True, hide_index=True)

    st.markdown("---")
    if st.button("🔄 Play Again", type="primary"):
        del st.session_state["game_state"]
        st.session_state["game_over"] = False
        if "last_round_result" in st.session_state:
            del st.session_state["last_round_result"]
        st.rerun()


def _plot_game_progress(state: GameState):
    """P&L, spot, and vol chart."""
    if not state.pnl_history:
        return

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=["Cumulative P&L", "Spot", "Implied Vol"],
        vertical_spacing=0.08, shared_xaxes=True,
    )

    rounds = [h["round"] for h in state.pnl_history]
    fig.add_trace(go.Scatter(x=rounds, y=[h["cumulative"] for h in state.pnl_history],
                             mode="lines+markers", name="P&L", line=dict(color="#636EFA")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(go.Scatter(x=rounds, y=[h["spot"] for h in state.pnl_history],
                             mode="lines", name="Spot", line=dict(color="#00CC96")), row=2, col=1)

    fig.add_trace(go.Scatter(x=rounds, y=[h["vol"] for h in state.pnl_history],
                             mode="lines", name="IV %", line=dict(color="#AB63FA")), row=3, col=1)

    fig.update_layout(height=450, showlegend=False)
    fig.update_xaxes(title_text="Week", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True)
