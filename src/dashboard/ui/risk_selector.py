"""Risk factor selection UI component."""

import streamlit as st
from dashboard.portfolio.builder import GREEKS


def render_risk_selector() -> dict:
    """Render the risk selector sidebar and return user's target profile.

    Returns:
        dict mapping greek name → "long"/"short"/"neutral"
    """
    st.sidebar.header("🎯 Target Risk Exposure")

    st.sidebar.markdown(
        "Select which risks you want exposure to. "
        "The system will build a portfolio with these exposures "
        "and hedge everything else."
    )

    # Asset class selection
    st.sidebar.subheader("Asset Classes")
    asset_classes = {}
    asset_classes["equity"] = st.sidebar.checkbox("Equity (SPY options)", value=True)
    asset_classes["fx"] = st.sidebar.checkbox("FX (EUR/USD options)", value=True)
    asset_classes["rates"] = st.sidebar.checkbox("Rates (USD swaptions)", value=False)

    # Greek selection
    st.sidebar.subheader("Greek Exposures")
    st.sidebar.caption("Choose direction for each risk factor")

    greek_descriptions = {
        "delta": "Δ Delta — directional spot exposure",
        "gamma": "Γ Gamma — convexity / acceleration",
        "vega": "ν Vega — volatility exposure",
        "theta": "Θ Theta — time decay",
        "rho": "ρ Rho — interest rate sensitivity",
        "vanna": "Vanna — spot-vol cross sensitivity",
        "volga": "Volga — vol-of-vol sensitivity",
    }

    target_greeks = {}
    for g in GREEKS:
        col1, col2 = st.sidebar.columns([3, 2])
        with col1:
            st.markdown(f"**{greek_descriptions[g]}**")
        with col2:
            direction = st.selectbox(
                f"{g}", ["neutral", "long", "short"],
                key=f"greek_{g}", label_visibility="collapsed",
            )
        target_greeks[g] = direction

    # Ticker selection
    st.sidebar.subheader("Instruments")
    ticker = st.sidebar.text_input("Equity ticker", value="SPY")
    fx_pair = st.sidebar.selectbox("FX pair", ["EURUSD", "USDJPY", "GBPUSD"])

    return {
        "target_greeks": target_greeks,
        "asset_classes": asset_classes,
        "ticker": ticker,
        "fx_pair": fx_pair,
    }
