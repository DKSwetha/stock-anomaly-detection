"""
src/data/ingest.py
------------------
Downloads historical OHLCV stock data for a basket of tickers
from Yahoo Finance and saves each as a CSV in data/raw/.
"""

import os
import yfinance as yf
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────
# A diverse basket spanning tech, finance, energy, retail, healthcare, etc.
# Training on varied sectors helps the model learn general "normal sequence"
# behavior instead of memorizing one company's price levels.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # Tech
    "JPM", "BAC", "GS",                         # Finance
    "XOM", "CVX",                               # Energy
    "WMT", "TGT", "COST",                       # Retail
    "JNJ", "PFE", "UNH",                        # Healthcare
    "DIS", "NKE",                               # Consumer
    "TSLA", "NVDA"                              # High-volatility tech
]

START   = "2018-01-01"
END     = "2024-01-01"
RAW_DIR = "data/raw"
# ───────────────────────────────────────────────────────────────────────────────


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from Yahoo Finance.

    Returns:
        DataFrame with columns [Open, High, Low, Close, Volume], or
        an empty DataFrame if the download failed.
    """
    print(f"[INFO] Downloading {ticker} ({start} -> {end}) ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        print(f"[WARN] No data returned for '{ticker}'. Skipping.")
        return df

    # Flatten multi-level columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    print(f"[INFO] {ticker}: {len(df)} rows")
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, raw_dir: str = RAW_DIR) -> str:
    """Save a single ticker's DataFrame to data/raw/<TICKER>.csv"""
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{ticker}.csv")
    df.to_csv(path)
    return path


def load_raw_data(ticker: str, raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Load a previously saved raw CSV back into a DataFrame."""
    path = os.path.join(raw_dir, f"{ticker}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw data found at '{path}'. Run download first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


def download_all(tickers: list[str] = TICKERS, start: str = START, end: str = END) -> list[str]:
    """
    Download and save raw data for every ticker in the basket.

    Returns:
        List of tickers that were successfully downloaded and saved.
    """
    successful = []

    for ticker in tickers:
        df = download_stock_data(ticker, start, end)
        if df.empty:
            continue
        save_raw_data(df, ticker)
        successful.append(ticker)

    print(f"\n[INFO] Successfully downloaded {len(successful)}/{len(tickers)} tickers")
    print(f"[INFO] Tickers: {successful}")
    return successful


# ── Quick smoke-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    successful = download_all()

    # Sanity check on one ticker
    if successful:
        sample = load_raw_data(successful[0])
        print(f"\n[CHECK] Sample ticker: {successful[0]}")
        print(sample.describe().round(2))
