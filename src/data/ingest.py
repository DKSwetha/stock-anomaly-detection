"""
src/data/ingest.py
------------------
Downloads historical OHLCV stock data from Yahoo Finance
and saves it as a CSV file in the data/raw/ directory.
"""

import os
import yfinance as yf
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────
TICKER    = "AAPL"          # Stock symbol to download
START     = "2018-01-01"    # Training data start date
END       = "2024-01-01"    # Training data end date
RAW_DIR   = "data/raw"      # Folder where raw CSVs are saved
# ───────────────────────────────────────────────────────────────────────────────


def download_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download OHLCV data for a given ticker from Yahoo Finance.
    """
    print(f"[INFO] Downloading data for {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'.")

    # Flatten multi-level columns if present (newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only the 5 core OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    print(f"[INFO] Downloaded {len(df)} rows  |  {df.shape[1]} features")
    print(df.tail(3))
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, raw_dir: str = RAW_DIR) -> str:
    """
    Save the DataFrame to data/raw/<TICKER>.csv
    """
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{ticker}.csv")
    df.to_csv(path)
    print(f"[INFO] Raw data saved → {path}")
    return path


def load_raw_data(ticker: str, raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """
    Load previously saved raw CSV back into a DataFrame.
    """
    path = os.path.join(raw_dir, f"{ticker}.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No raw data found at '{path}'. Run download first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"

    print(f"[INFO] Loaded {len(df)} rows from {path}")
    return df


# ── Quick smoke-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    df = download_stock_data(TICKER, START, END)
    save_raw_data(df, TICKER)

    # Verify the round-trip load works
    df_loaded = load_raw_data(TICKER)
    print("\n[CHECK] Data types:")
    print(df_loaded.dtypes)
    print("\n[CHECK] Missing values:")
    print(df_loaded.isnull().sum())
    print("\n[CHECK] Basic stats:")
    print(df_loaded.describe().round(2))
