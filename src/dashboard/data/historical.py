"""Historical SPX daily returns for realistic simulation paths.

Downloads once from yfinance and caches to disk permanently.
Provides daily returns and realized vol for sampling.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parents[3] / "data" / "cache"
CACHE_FILE = CACHE_DIR / "spx_history.parquet"
META_FILE = CACHE_DIR / "spx_history_meta.json"


def get_spx_returns(years: int = 5) -> pd.DataFrame:
    """Get daily SPX returns. Downloads once, then uses cache forever.

    Returns DataFrame with columns: date, close, return_pct, realized_vol_20d
    """
    if CACHE_FILE.exists():
        return pd.read_parquet(CACHE_FILE)

    import yfinance as yf
    tk = yf.Ticker("^GSPC")
    hist = tk.history(period=f"{years}y")
    if hist.empty:
        raise ValueError("Cannot fetch SPX history")

    df = pd.DataFrame({
        "date": hist.index,
        "close": hist["Close"].values,
    })
    df["return_pct"] = df["close"].pct_change()
    df["realized_vol_20d"] = df["return_pct"].rolling(20).std()
    df = df.dropna().reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_FILE)
    META_FILE.write_text(json.dumps({"timestamp": time.time(), "rows": len(df)}))
    return df


def sample_historical_moves(n_days: int, seed: int | None = None) -> np.ndarray:
    """Sample n_days of returns from historical SPX data (with replacement).

    Returns array of daily return fractions (e.g., 0.01 = +1%).
    """
    df = get_spx_returns()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=n_days, replace=True)
    return df["return_pct"].iloc[indices].values
