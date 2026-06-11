"""
src/data/preprocess.py
----------------------
Normalizes OHLCV data with MinMaxScaler and creates
overlapping sliding windows for the LSTM Autoencoder.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_SIZE    = 30        # Number of time steps per sequence
TRAIN_RATIO    = 0.8       # 80% train, 20% test
PROCESSED_DIR  = "data/processed"
SCALER_DIR     = "data/scalers"
FEATURES       = ["Open", "High", "Low", "Close", "Volume"]
# ───────────────────────────────────────────────────────────────────────────────


def normalize(df: pd.DataFrame, ticker: str, fit: bool = True) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize the OHLCV DataFrame using MinMaxScaler.

    Args:
        df     : Raw OHLCV DataFrame
        ticker : Used for saving/loading the scaler
        fit    : If True, fit a new scaler. If False, load existing one.

    Returns:
        (scaled_array, scaler) — scaled_array shape: (n_rows, n_features)
    """
    os.makedirs(SCALER_DIR, exist_ok=True)
    scaler_path = os.path.join(SCALER_DIR, f"{ticker}_scaler.pkl")

    data = df[FEATURES].values

    if fit:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)
        print(f"[INFO] Scaler fitted and saved → {scaler_path}")
    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at '{scaler_path}'. Run with fit=True first.")
        scaler = joblib.load(scaler_path)
        scaled = scaler.transform(data)
        print(f"[INFO] Scaler loaded from {scaler_path}")

    return scaled, scaler


def create_windows(scaled: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Convert a 2D scaled array into overlapping 3D windows.

    Example:
        Input  shape: (1000, 5)
        Output shape: (971, 30, 5)   ← 1000 - 30 + 1 = 971 windows

    Args:
        scaled      : 2D array (n_rows, n_features)
        window_size : Number of time steps per window

    Returns:
        3D array of shape (n_windows, window_size, n_features)
    """
    windows = []
    for i in range(len(scaled) - window_size + 1):
        windows.append(scaled[i : i + window_size])
    windows = np.array(windows)
    print(f"[INFO] Created {windows.shape[0]} windows of shape {windows.shape[1:]}")
    return windows


def train_test_split_windows(windows: np.ndarray, train_ratio: float = TRAIN_RATIO):
    """
    Split windows into train and test sets (no shuffle — time order matters).

    Args:
        windows     : 3D array (n_windows, window_size, n_features)
        train_ratio : Fraction for training

    Returns:
        (X_train, X_test)
    """
    split = int(len(windows) * train_ratio)
    X_train = windows[:split]
    X_test  = windows[split:]
    print(f"[INFO] Train windows: {len(X_train)}  |  Test windows: {len(X_test)}")
    return X_train, X_test


def save_processed(X_train: np.ndarray, X_test: np.ndarray, ticker: str):
    """Save processed numpy arrays to disk."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(os.path.join(PROCESSED_DIR, f"{ticker}_X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, f"{ticker}_X_test.npy"),  X_test)
    print(f"[INFO] Processed arrays saved to {PROCESSED_DIR}/")


def load_processed(ticker: str):
    """Load processed numpy arrays from disk."""
    X_train = np.load(os.path.join(PROCESSED_DIR, f"{ticker}_X_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED_DIR, f"{ticker}_X_test.npy"))
    print(f"[INFO] Loaded X_train {X_train.shape}  |  X_test {X_test.shape}")
    return X_train, X_test


def run_preprocessing(ticker: str = "AAPL"):
    """
    Full preprocessing pipeline:
    load raw CSV → normalize → window → split → save
    """
    from ingest import load_raw_data   # local import to avoid circular deps

    df        = load_raw_data(ticker)
    scaled, _ = normalize(df, ticker, fit=True)
    windows   = create_windows(scaled)
    X_train, X_test = train_test_split_windows(windows)
    save_processed(X_train, X_test, ticker)

    print("\n Preprocessing complete!")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    return X_train, X_test


# ── Quick smoke-test when run directly ─────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing("AAPL")
