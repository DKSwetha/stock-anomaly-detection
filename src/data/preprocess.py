"""
src/data/preprocess.py
----------------------
For each ticker: normalize with its own MinMaxScaler and create
sliding windows. Then combine all tickers' windows into one
shared train/test dataset for the generalized model.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

from .ingest import TICKERS, load_raw_data


# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_SIZE   = 30
TRAIN_RATIO   = 0.8
PROCESSED_DIR = "data/processed"
SCALER_DIR    = "data/scalers"
FEATURES      = ["Open", "High", "Low", "Close", "Volume"]
# ───────────────────────────────────────────────────────────────────────────────


def normalize(df: pd.DataFrame, ticker: str, fit: bool = True) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize one ticker's OHLCV DataFrame with its own MinMaxScaler.

    Each ticker gets its own scaler because price ranges differ wildly
    (e.g. AAPL ~$190 vs a stock trading at $20). Scaling per-ticker means
    the model learns the *shape* of normal sequences, not absolute price levels.
    """
    os.makedirs(SCALER_DIR, exist_ok=True)
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")

    data = df[FEATURES].values

    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at '{scaler_path}'. Run with fit=True first.")
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(data)

    return scaled, scaler


def create_windows(scaled: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Convert a 2D scaled array into overlapping 3D windows.

    Input  shape: (n_rows, n_features)
    Output shape: (n_rows - window_size + 1, window_size, n_features)
    """
    if len(scaled) < window_size:
        return np.empty((0, window_size, scaled.shape[1]))

    windows = []
    for i in range(len(scaled) - window_size + 1):
        windows.append(scaled[i: i + window_size])
    return np.array(windows)


def train_test_split_windows(windows: np.ndarray, train_ratio: float = TRAIN_RATIO):
    """Split one ticker's windows into train/test (time-ordered, no shuffle)."""
    split = int(len(windows) * train_ratio)
    return windows[:split], windows[split:]


def save_processed(X_train: np.ndarray, X_test: np.ndarray, name: str = "combined"):
    """Save processed numpy arrays to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, f"{name}_X_test.npy"), X_test)
    print(f"[INFO] Saved {name}_X_train.npy {X_train.shape} and {name}_X_test.npy {X_test.shape}")


def load_processed(name: str = "combined"):
    """Load combined processed numpy arrays from disk."""
    X_train = np.load(os.path.join(PROCESSED_DIR, f"{name}_X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, f"{name}_X_test.npy"))
    print(f"[INFO] Loaded X_train {X_train.shape} | X_test {X_test.shape}")
    return X_train, X_test


def run_preprocessing(tickers: list[str] = None, window_size: int = WINDOW_SIZE):
    """
    Full multi-ticker preprocessing pipeline:

    For each ticker:
        load raw CSV -> normalize (own scaler) -> create windows -> split train/test

    Then concatenate all tickers' train windows into one combined X_train,
    and all tickers' test windows into one combined X_test.
    """
    if tickers is None:
        tickers = TICKERS

    all_train, all_test = [], []
    summary = []

    for ticker in tickers:
        try:
            df = load_raw_data(ticker)
        except FileNotFoundError:
            print(f"[WARN] No raw data for {ticker}, skipping.")
            continue

        scaled, _ = normalize(df, ticker, fit=True)
        windows = create_windows(scaled, window_size)

        if len(windows) == 0:
            print(f"[WARN] Not enough rows for {ticker} to form a window, skipping.")
            continue

        X_train, X_test = train_test_split_windows(windows)
        all_train.append(X_train)
        all_test.append(X_test)
        summary.append((ticker, X_train.shape[0], X_test.shape[0]))

    # Combine all tickers into one shared dataset
    X_train_combined = np.concatenate(all_train, axis=0)
    X_test_combined = np.concatenate(all_test, axis=0)

    # Shuffle training windows so the model doesn't see one ticker
    # at a time in a fixed order during training
    rng = np.random.default_rng(seed=42)
    rng.shuffle(X_train_combined)

    save_processed(X_train_combined, X_test_combined, name="combined")

    print("\n[SUMMARY] Per-ticker window counts (train / test):")
    for ticker, n_train, n_test in summary:
        print(f"   {ticker:6s} -> train: {n_train:4d} | test: {n_test:4d}")

    print("\nPreprocessing complete!")
    print(f"   X_train_combined : {X_train_combined.shape}")
    print(f"   X_test_combined  : {X_test_combined.shape}")
    print(f"   Tickers used     : {[t for t, _, _ in summary]}")

    return X_train_combined, X_test_combined


# ── Quick smoke-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing()
