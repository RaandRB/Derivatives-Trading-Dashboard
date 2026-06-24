"""Equity option chain data fetching via yfinance.

Provides live (delayed ~15min) option chains for US equities/ETFs.
Caches results to disk with configurable TTL to avoid hammering Yahoo.
"""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).parents[3] / "data" / "cache"
DEFAULT_TTL = 900  # 15 minutes


def get_option_chain(ticker: str, ttl: int = DEFAULT_TTL) -> pd.DataFrame:
    """Fetch full option chain for a ticker across all available expiries.

    Returns DataFrame with columns:
        strike, expiry, option_type, bid, ask, mid, impliedVolatility,
        volume, openInterest, lastPrice, inTheMoney
    """
    cached = _load_cache(ticker, ttl)
    if cached is not None:
        return cached

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise ValueError(f"No options available for {ticker}")

    frames = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["expiry"] = pd.Timestamp(exp)
            df["option_type"] = opt_type
            frames.append(df)

    result = pd.concat(frames, ignore_index=True)
    result["mid"] = (result["bid"] + result["ask"]) / 2
    result = result.rename(columns={"impliedVolatility": "iv"})

    # Keep only useful columns
    cols = [
        "strike", "expiry", "option_type", "bid", "ask", "mid",
        "iv", "volume", "openInterest", "lastPrice", "inTheMoney",
    ]
    result = result[[c for c in cols if c in result.columns]]
    result = result.dropna(subset=["strike", "iv"])
    result = result[result["iv"] > 0]

    _save_cache(ticker, result)
    return result


def get_spot(ticker: str) -> float:
    """Current spot price."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1d")
    if hist.empty:
        raise ValueError(f"Cannot fetch spot for {ticker}")
    return float(hist["Close"].iloc[-1])


def _cache_path(ticker: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"options_{ticker.upper()}.parquet"


def _meta_path(ticker: str) -> Path:
    return CACHE_DIR / f"options_{ticker.upper()}_meta.json"


def _load_cache(ticker: str, ttl: int) -> pd.DataFrame | None:
    meta = _meta_path(ticker)
    path = _cache_path(ticker)
    if not meta.exists() or not path.exists():
        return None
    with open(meta) as f:
        info = json.load(f)
    if time.time() - info["timestamp"] > ttl:
        return None
    return pd.read_parquet(path)


def _save_cache(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cache_path(ticker))
    with open(_meta_path(ticker), "w") as f:
        json.dump({"timestamp": time.time()}, f)
