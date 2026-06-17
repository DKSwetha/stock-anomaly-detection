"""
src/inference/predict.py
-------------------------
Core inference logic for the anomaly detection API.

Given a ticker symbol:
  1. Fetch recent OHLCV data via yfinance
  2. Fit a fresh MinMaxScaler on this data
  3. Build sliding windows
  4. Run the trained LSTM Autoencoder
  5. Flag anomalies using z-score thresholding
  6. Compute per-feature error breakdown for anomalous windows

Anomaly detection strategy:
  A window is flagged as anomalous if BOTH conditions are true:
    a) Its reconstruction error is more than Z_THRESHOLD standard
       deviations above the mean of this ticker's own errors
    b) Its raw error exceeds a minimum floor (MIN_ERROR_FLOOR)
       so statistically-outlier-but-tiny errors aren't flagged

  This gives a natural, data-driven anomaly count that varies
  per ticker rather than always returning a fixed quota.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.ingest import download_stock_data
from src.model.autoencoder import LSTMAutoencoder


# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_SIZE    = 30
N_FEATURES     = 5
HIDDEN_SIZE    = 64
NUM_LAYERS     = 1
FEATURES       = ["Open", "High", "Low", "Close", "Volume"]

MODEL_PATH     = "models/lstm_autoencoder.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Z_THRESHOLD     = 1.5
LOOKBACK_DAYS  = 90     # calendar days to fetch (~60 trading days)
MIN_ERROR_FLOOR = 0.008 # never flag windows with very low absolute error
# ───────────────────────────────────────────────────────────────────────────────

_model     = None


def load_model() -> LSTMAutoencoder:
    """Load the trained model once and cache it in memory."""
    global _model
    if _model is None:
        model = LSTMAutoencoder(
            n_features=N_FEATURES,
            window_size=WINDOW_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        _model = model
    return _model


def fetch_recent_data(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Fetch the most recent `lookback_days` of OHLCV data for a ticker.

    Raises:
        ValueError: if ticker is invalid or not enough data exists.
    """
    end   = pd.Timestamp.today()
    start = end - pd.Timedelta(days=lookback_days)

    df = download_stock_data(
        ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # Drop rows with missing values (market holidays, data gaps)
    df = df.dropna()

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Only {len(df)} trading days available for '{ticker}', "
            f"need at least {WINDOW_SIZE}."
        )

    return df


def scale_data(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Fit a fresh MinMaxScaler on this ticker's recent data.

    The model never memorized absolute price levels — each training ticker
    was scaled independently — so fitting a fresh scaler at inference time
    is consistent with training and allows any ticker to be analyzed.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)
    return scaled, scaler


def create_windows(scaled: np.ndarray, window_size: int = WINDOW_SIZE) -> np.ndarray:
    """Create overlapping 30-day windows from scaled data."""
    windows = []
    for i in range(len(scaled) - window_size + 1):
        windows.append(scaled[i: i + window_size])
    return np.array(windows)


def run_inference(model: LSTMAutoencoder, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on all windows.

    Returns:
        reconstructions: (n_windows, window_size, n_features)
        errors         : per-window MSE, shape (n_windows,)
    """
    with torch.no_grad():
        x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
        reconstructions = model(x).cpu().numpy()

    errors = np.mean((reconstructions - windows) ** 2, axis=(1, 2))
    return reconstructions, errors


def detect_anomalies(errors: np.ndarray) -> tuple[np.ndarray, float, str]:
    mean = np.mean(errors)
    std  = np.std(errors)
    threshold = mean + Z_THRESHOLD * std
    is_anomaly = (errors > threshold) & (errors > MIN_ERROR_FLOOR)
    return is_anomaly, float(threshold), "z-score (mean + 1.5*std) with min error floor"

def per_feature_error(window: np.ndarray, reconstruction: np.ndarray) -> dict:
    """
    Compute per-feature reconstruction error for a single window.

    Returns dict: {"Open": float, "High": float, ..., "Volume": float}
    The top_feature with highest error is the main driver of the anomaly.
    """
    per_feat_mse = np.mean((reconstruction - window) ** 2, axis=0)
    return {feat: float(err) for feat, err in zip(FEATURES, per_feat_mse)}


def analyze_ticker(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> dict:
    """
    Full end-to-end analysis for a single ticker.

    Returns a dict with:
        ticker               : symbol analyzed
        dates                : date for each window's last day
        prices               : OHLCV values for those dates (for plotting)
        reconstruction_errors: per-window error score
        error_mean           : mean error across all windows
        error_std            : std of errors across all windows
        threshold            : z-score based cutoff (mean + 1.5*std)
        threshold_method     : z-score thresholding strategy
        is_anomaly           : bool per window
        anomaly_details      : per-feature breakdown for anomalous windows
        n_anomalies          : count of flagged windows
    """
    model      = load_model()
    df         = fetch_recent_data(ticker, lookback_days)
    scaled, _  = scale_data(df)
    windows    = create_windows(scaled)

    if len(windows) == 0:
        raise ValueError(f"Not enough data to form windows for '{ticker}'.")

    reconstructions, errors = run_inference(model, windows)
    is_anomaly, threshold, method = detect_anomalies(errors)

    print(f"[INFO] Error stats — mean: {np.mean(errors):.6f}, "
          f"std: {np.std(errors):.6f}, threshold: {threshold:.6f}")
    print(f"[INFO] Anomalies: {is_anomaly.sum()} / {len(windows)} windows")

    # Each window's label date is the last day in that window
    window_end_dates = df.index[WINDOW_SIZE - 1:]

    anomaly_details = {}
    for i, anomalous in enumerate(is_anomaly):
        if anomalous:
            date_str  = window_end_dates[i].strftime("%Y-%m-%d")
            breakdown = per_feature_error(windows[i], reconstructions[i])
            top_feat  = max(breakdown, key=breakdown.get)
            anomaly_details[date_str] = {
                "reconstruction_error": float(errors[i]),
                "feature_errors"      : breakdown,
                "top_feature"         : top_feat,
            }

    return {
        "ticker"              : ticker,
        "dates"               : [d.strftime("%Y-%m-%d") for d in window_end_dates],
        "prices"              : df.iloc[WINDOW_SIZE - 1:][FEATURES].to_dict(orient="list"),
        "reconstruction_errors": errors.tolist(),
        "error_mean"          : float(np.mean(errors)),
        "error_std"           : float(np.std(errors)),
        "threshold"           : threshold,
        "threshold_method"    : method,
        "is_anomaly"          : is_anomaly.tolist(),
        "anomaly_details"     : anomaly_details,
        "n_anomalies"         : int(is_anomaly.sum()),
    }


# ── Quick smoke-test when run directly ──────────────────────────────────────────
if __name__ == "__main__":
    for ticker in ["MSFT", "META", "DIS"]:
        print(f"\n{'='*50}")
        print(f"[INFO] Analyzing {ticker} ...")
        result = analyze_ticker(ticker)

        print(f"  Windows analyzed  : {len(result['dates'])}")
        print(f"  Error mean        : {result['error_mean']:.6f}")
        print(f"  Error std         : {result['error_std']:.6f}")
        print(f"  Threshold (z-score): {result['threshold']:.6f}")
        print(f"  Anomalies found   : {result['n_anomalies']}")

        if result["anomaly_details"]:
            print("  Anomaly details:")
            for date, details in result["anomaly_details"].items():
                print(f"    {date}: error={details['reconstruction_error']:.6f}, "
                      f"top_feature={details['top_feature']}")
        else:
            print("  No anomalies detected.")
