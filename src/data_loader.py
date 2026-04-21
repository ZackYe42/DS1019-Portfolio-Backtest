"""
Data loader for S&P 100 stock prices.

Downloads daily adjusted close prices from Yahoo Finance, caches locally
as parquet, and provides a clean interface for downstream backtest code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# S&P 100 tickers (as of 2024).  Static list to keep the project reproducible.
# A couple of tickers from the official index have been dropped because they
# either delisted or have incomplete history across 2014-2024 (e.g. META IPO'd
# as FB, DD split, etc.).  This gives us 95 tickers with clean 10-year history.
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT",  "ACN",  "ADBE", "AIG",  "AMD",  "AMGN", "AMT",  "AMZN",
    "AVGO", "AXP",  "BA",   "BAC",  "BK",   "BLK",  "BMY",  "BRK-B","C",    "CAT",
    "CHTR", "CL",   "CMCSA","COF",  "COP",  "COST", "CRM",  "CSCO", "CVS",  "CVX",
    "DE",   "DHR",  "DIS",  "DUK",  "EMR",  "F",    "FDX",  "GD",   "GE",   "GILD",
    "GM",   "GOOG", "GOOGL","GS",   "HD",   "HON",  "IBM",  "INTC", "JNJ",  "JPM",
    "KO",   "LIN",  "LLY",  "LMT",  "LOW",  "MA",   "MCD",  "MDLZ", "MDT",  "MET",
    "MMM",  "MO",   "MRK",  "MS",   "MSFT", "NEE",  "NFLX", "NKE",  "NVDA", "ORCL",
    "OXY",  "PEP",  "PFE",  "PG",   "PM",   "PYPL", "QCOM", "RTX",  "SBUX", "SCHW",
    "SO",   "SPG",  "T",    "TGT",  "TMO",  "TMUS", "TSLA", "TXN",  "UNH",  "UNP",
    "UPS",  "USB",  "V",    "VZ",   "WFC",  "WMT",  "XOM",
]

# Resolve paths relative to the project root, no matter where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
PRICES_FILE = DATA_RAW / "prices.parquet"
RETURNS_FILE = DATA_PROCESSED / "returns.parquet"


def download_prices(
    tickers: list[str] = SP100_TICKERS,
    start: str = "2014-01-01",
    end: str = "2024-12-31",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted-close prices from Yahoo Finance and cache to parquet.

    Returns a DataFrame with a DatetimeIndex and one column per ticker.
    Tickers with more than 5% missing data are dropped (printed as a warning).
    """
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    if PRICES_FILE.exists() and not force_refresh:
        print(f"[data_loader] loading cached prices from {PRICES_FILE}")
        return pd.read_parquet(PRICES_FILE)

    print(f"[data_loader] downloading {len(tickers)} tickers from yfinance...")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,   # uses adjusted close as "Close"
        progress=True,
        threads=True,
    )

    # yfinance returns a MultiIndex on columns when given multiple tickers.
    # We want just the Close (adjusted) panel.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers[:1]

    # Drop tickers that are too incomplete
    missing_frac = prices.isna().mean()
    bad = missing_frac[missing_frac > 0.05].index.tolist()
    if bad:
        print(f"[data_loader] dropping {len(bad)} tickers with >5% missing: {bad}")
        prices = prices.drop(columns=bad)

    # Forward-fill small gaps (e.g. single-day trading halts), then drop any remaining NaN rows
    prices = prices.ffill().dropna(how="any")

    prices.to_parquet(PRICES_FILE)
    print(f"[data_loader] saved {prices.shape} to {PRICES_FILE}")
    return prices


def compute_log_returns(prices: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """
    Compute daily log returns from a price DataFrame and cache to parquet.
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    if RETURNS_FILE.exists() and not force_refresh:
        print(f"[data_loader] loading cached returns from {RETURNS_FILE}")
        return pd.read_parquet(RETURNS_FILE)

    returns = np.log(prices / prices.shift(1)).dropna(how="any")
    returns.to_parquet(RETURNS_FILE)
    print(f"[data_loader] saved {returns.shape} to {RETURNS_FILE}")
    return returns


def load_prices(force_refresh: bool = False) -> pd.DataFrame:
    """Public entry point.  Returns cached prices, downloading if needed."""
    return download_prices(force_refresh=force_refresh)


def load_returns(force_refresh: bool = False) -> pd.DataFrame:
    """Public entry point.  Returns cached log returns, computing if needed."""
    prices = load_prices(force_refresh=force_refresh)
    return compute_log_returns(prices, force_refresh=force_refresh)


if __name__ == "__main__":
    # Running `python -m src.data_loader` downloads and caches everything.
    prices = load_prices()
    returns = load_returns()

    print("\n=== Summary ===")
    print(f"Prices  : {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"Date    : {prices.index.min().date()}  ->  {prices.index.max().date()}")
    print(f"Returns : {returns.shape}")
    print(f"\nFirst 5 tickers:\n{prices.iloc[:3, :5]}")