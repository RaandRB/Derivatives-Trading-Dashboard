"""FX spot data and implied volatility surface.

Spot: fetched live from yfinance.
Vol surface: bundled representative quotes in delta-space convention
(ATM DNS, 25-delta risk reversal, 25-delta butterfly) — the standard
quoting convention used by FX options desks globally.

NOTE ON MARKET PRACTICE:
FX options are quoted in delta-space, not strike-space. The ATM convention
is typically "delta-neutral straddle" (DNS), where the ATM strike is chosen
such that the straddle is delta-neutral. This differs from equity ATM
(which is just spot or forward). We implement the proper FX convention here.

The bundled vol quotes are representative values consistent with recent
market conditions. In production, these would come from a live feed
(Bloomberg FXGO, Refinitiv, or direct bank pricing).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).parents[3] / "data"

# Supported pairs and their yfinance symbols
FX_PAIRS = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
}


def get_spot(pair: str) -> float:
    """Fetch current FX spot rate."""
    symbol = FX_PAIRS.get(pair.upper())
    if not symbol:
        raise ValueError(f"Unsupported pair: {pair}. Available: {list(FX_PAIRS)}")
    tk = yf.Ticker(symbol)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Cannot fetch spot for {pair}")
    return float(hist["Close"].iloc[-1])


def get_vol_surface(pair: str) -> dict:
    """Load FX vol surface in delta-space.

    Returns dict with structure:
        {expiry_label: {"atm": vol, "rr25": vol, "bf25": vol, "rr10": vol, "bf10": vol}}

    Where:
        atm = ATM delta-neutral straddle vol
        rr25 = 25-delta risk reversal (call vol - put vol)
        bf25 = 25-delta butterfly ((call vol + put vol)/2 - atm)
        rr10 = 10-delta risk reversal
        bf10 = 10-delta butterfly

    All vols expressed as decimals (e.g., 0.08 = 8%).
    """
    path = DATA_DIR / "fx_vols.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if pair.upper() in data:
            return data[pair.upper()]
    return _default_vol_surface(pair)


def _default_vol_surface(pair: str) -> dict:
    """Representative FX vol quotes.

    These approximate mid-2025 market conditions:
    - EURUSD: moderate vol, slight put skew (USD strength bias)
    - USDJPY: higher vol, significant put skew (yen carry unwinds)
    - GBPUSD: moderate vol, mild skew
    """
    surfaces = {
        "EURUSD": {
            "1W": {"atm": 0.072, "rr25": -0.004, "bf25": 0.003, "rr10": -0.009, "bf10": 0.009},
            "1M": {"atm": 0.078, "rr25": -0.006, "bf25": 0.004, "rr10": -0.013, "bf10": 0.012},
            "3M": {"atm": 0.082, "rr25": -0.008, "bf25": 0.005, "rr10": -0.016, "bf10": 0.014},
            "6M": {"atm": 0.085, "rr25": -0.009, "bf25": 0.005, "rr10": -0.018, "bf10": 0.015},
            "1Y": {"atm": 0.088, "rr25": -0.010, "bf25": 0.006, "rr10": -0.020, "bf10": 0.017},
        },
        "USDJPY": {
            "1W": {"atm": 0.095, "rr25": -0.012, "bf25": 0.005, "rr10": -0.025, "bf10": 0.015},
            "1M": {"atm": 0.102, "rr25": -0.015, "bf25": 0.006, "rr10": -0.030, "bf10": 0.018},
            "3M": {"atm": 0.108, "rr25": -0.018, "bf25": 0.007, "rr10": -0.035, "bf10": 0.020},
            "6M": {"atm": 0.112, "rr25": -0.020, "bf25": 0.008, "rr10": -0.038, "bf10": 0.022},
            "1Y": {"atm": 0.115, "rr25": -0.022, "bf25": 0.009, "rr10": -0.040, "bf10": 0.025},
        },
        "GBPUSD": {
            "1W": {"atm": 0.068, "rr25": -0.003, "bf25": 0.003, "rr10": -0.007, "bf10": 0.008},
            "1M": {"atm": 0.074, "rr25": -0.005, "bf25": 0.004, "rr10": -0.010, "bf10": 0.010},
            "3M": {"atm": 0.078, "rr25": -0.006, "bf25": 0.004, "rr10": -0.013, "bf10": 0.012},
            "6M": {"atm": 0.081, "rr25": -0.007, "bf25": 0.005, "rr10": -0.015, "bf10": 0.013},
            "1Y": {"atm": 0.084, "rr25": -0.008, "bf25": 0.005, "rr10": -0.017, "bf10": 0.015},
        },
    }
    pair_upper = pair.upper()
    if pair_upper in surfaces:
        return surfaces[pair_upper]
    # Default to EURUSD-like surface
    return surfaces["EURUSD"]


def expiry_to_years(label: str) -> float:
    """Convert expiry label to year fraction."""
    mapping = {"1W": 1/52, "2W": 2/52, "1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0}
    return mapping.get(label, 0.25)
