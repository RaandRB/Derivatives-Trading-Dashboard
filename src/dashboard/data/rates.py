"""US Treasury yield curve (bundled) + bundled swaption vol data.

Yield curve: bundled snapshot of Treasury constant maturity rates.
Swaption vols: bundled snapshot from public sources (ICE/CME indicative data).

NOTE: The bundled swaption vols are representative snapshots. In production,
desks use live feeds from Bloomberg/Refinitiv. The yield curve uses Treasury
rates as a proxy for the swap curve (typically ~5-15bp spread for USD due to
credit/liquidity differences). A production system would use OIS-discounted
swap rates.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parents[3] / "data"

# Tenor in years for each series
TENOR_YEARS = {
    "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0, "3Y": 3.0,
    "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}


def get_yield_curve() -> pd.DataFrame:
    """Get US Treasury yield curve (bundled snapshot).

    Returns DataFrame with columns: tenor_label, tenor_years, rate (as decimal).
    """
    return _load_bundled_curve()


def _load_bundled_curve() -> pd.DataFrame:
    """Representative USD yield curve (June 2025 snapshot)."""
    rates = {
        "3M": 0.0435, "6M": 0.0428, "1Y": 0.0410, "2Y": 0.0395,
        "3Y": 0.0388, "5Y": 0.0385, "7Y": 0.0390, "10Y": 0.0400,
        "20Y": 0.0435, "30Y": 0.0445,
    }
    rows = [
        {"tenor_label": k, "tenor_years": TENOR_YEARS[k], "rate": v}
        for k, v in rates.items()
    ]
    return pd.DataFrame(rows)


def get_swaption_vols() -> dict:
    """Load bundled swaption vol cube.

    Returns dict with structure:
        {expiry_tenor: {underlying_tenor: {strike_offset: vol}}}

    Strike offsets are in bp relative to ATM forward swap rate.
    Vols are normal (bp) vols — standard for USD swaptions post-2008.

    SOURCE: Representative values based on publicly available ICE/CME
    indicative swaption vol data. These are NOT live quotes.
    Last updated: 2025-06 (approximate).
    """
    path = DATA_DIR / "swaption_vols.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return _default_swaption_vols()


def _default_swaption_vols() -> dict:
    """Representative USD swaption normal vol cube (bp/year).

    In production, swaption vols are quoted in either:
    - Normal (Bachelier) vol in bp — standard for USD/EUR since rates went negative
    - Lognormal (Black) vol in % — legacy, still used for some currencies

    We use normal vols here as that's current market practice for USD.
    """
    # ATM normal vols (bp/year) for expiry x underlying tenor
    # Rows: option expiry, Cols: underlying swap tenor
    atm = {
        "1M": {"1Y": 95, "2Y": 92, "5Y": 85, "10Y": 78, "30Y": 68},
        "3M": {"1Y": 100, "2Y": 97, "5Y": 88, "10Y": 80, "30Y": 70},
        "6M": {"1Y": 103, "2Y": 99, "5Y": 90, "10Y": 82, "30Y": 72},
        "1Y": {"1Y": 105, "2Y": 101, "5Y": 92, "10Y": 84, "30Y": 73},
        "2Y": {"1Y": 102, "2Y": 98, "5Y": 90, "10Y": 83, "30Y": 72},
        "5Y": {"1Y": 95, "2Y": 92, "5Y": 86, "10Y": 80, "30Y": 70},
        "10Y": {"1Y": 85, "2Y": 83, "5Y": 79, "10Y": 75, "30Y": 66},
    }
    # SABR parameters (representative) for smile generation
    # alpha ≈ ATM vol, beta = 0.5 (CEV, market standard for USD),
    # rho = -0.2 (negative skew typical), nu = 0.3 (vol of vol)
    sabr_params = {
        "beta": 0.5,
        "rho": -0.20,
        "nu": 0.30,
    }
    return {"atm_normal_vols_bp": atm, "sabr_params": sabr_params}
