"""
src/monitoring/monitor.py
--------------------------
Generates Evidently AI data drift reports comparing the
reference distribution (training data) against a live
ticker's recent data.

This answers the question: "Is the data coming into the model
today statistically similar to what it was trained on?"

If drift is detected, it means the model's learned notion of
"normal" may no longer apply to current market conditions —
a signal that retraining should be considered.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataSummaryPreset

from src.data.ingest import load_raw_data, download_stock_data, TICKERS
from src.data.preprocess import normalize, FEATURES

# ── Configuration ──────────────────────────────────────────────────────────────
REPORTS_DIR       = "monitoring/reports"
LOOKBACK_DAYS     = 90
# ───────────────────────────────────────────────────────────────────────────────


def build_reference_df(tickers: list[str] = None, sample_per_ticker: int = 100) -> pd.DataFrame:
    """
    Build a reference DataFrame from training data.

    Loads raw CSVs for each ticker, normalizes with its own scaler,
    and samples rows. The combined result represents the distribution
    of normal data the model was trained on.
    """
    if tickers is None:
        tickers = TICKERS

    dfs = []
    for ticker in tickers:
        try:
            df        = load_raw_data(ticker)
            scaled, _ = normalize(df, ticker, fit=False)
            scaled_df = pd.DataFrame(scaled, columns=FEATURES)
            n         = min(sample_per_ticker, len(scaled_df))
            dfs.append(scaled_df.sample(n, random_state=42))
        except Exception as e:
            print(f"[WARN] Skipping {ticker} for reference: {e}")

    if not dfs:
        raise RuntimeError("Could not build reference dataset — no ticker data found.")

    reference = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Reference dataset: {len(reference)} rows from {len(dfs)} tickers")
    return reference


def build_current_df(ticker: str, lookback_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Build a current DataFrame from recent live data for a ticker.

    Fetches recent OHLCV and normalizes with a fresh scaler —
    consistent with how inference handles unseen tickers.
    """
    end   = pd.Timestamp.today()
    start = end - pd.Timedelta(days=lookback_days)

    df = download_stock_data(
        ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if df.empty:
        raise ValueError(f"No recent data for '{ticker}'.")

    df     = df.dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FEATURES].values)
    current = pd.DataFrame(scaled, columns=FEATURES)

    print(f"[INFO] Current dataset ({ticker}): {len(current)} rows")
    return current


def generate_drift_report(ticker: str, output_dir: str = REPORTS_DIR) -> str:
    """
    Generate an Evidently HTML drift report for a ticker.

    Compares training distribution (reference) against recent live
    data (current) and saves the report to monitoring/reports/.

    Returns:
        Path to the saved HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)

    reference_df = build_reference_df()
    current_df   = build_current_df(ticker)

    data_definition = DataDefinition(
        numerical_columns=FEATURES,
    )

    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current   = Dataset.from_pandas(current_df, data_definition=data_definition)

    drift_metrics = [ValueDrift(column=col) for col in FEATURES]
    report = Report(metrics=[*drift_metrics, DataSummaryPreset()])

    snapshot = report.run(reference_data=reference, current_data=current)

    output_path = os.path.join(output_dir, f"{ticker}_drift_report.html")
    snapshot.save_html(output_path)
    print(f"[INFO] Drift report saved -> {output_path}")
    return output_path


def get_drift_summary(ticker: str) -> dict:
    """
    Run drift detection and return a summary dict for the dashboard.

    Returns:
        ticker         : symbol analyzed
        dataset_drift  : True if overall drift detected
        n_drifted_cols : number of features with drift
        feature_drift  : per-feature drift flag dict
        share_drifted  : fraction of features drifted (0.0 to 1.0)
    """
    reference_df = build_reference_df()
    current_df   = build_current_df(ticker)

    data_definition = DataDefinition(
        numerical_columns=FEATURES,
    )

    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current   = Dataset.from_pandas(current_df, data_definition=data_definition)

    drift_metrics = [ValueDrift(column=col) for col in FEATURES]
    report = Report(metrics=drift_metrics)
    snapshot = report.run(reference_data=reference, current_data=current)

    results      = snapshot.dict()
    feature_drift = {}
    for m in results["metrics"]:
        col = m["config"]["column"]
        p_value = m["value"]
        threshold = m["config"]["threshold"]
        feature_drift[col] = bool(p_value < threshold)

    n_drifted = sum(feature_drift.values())

    return {
        "ticker"        : ticker,
        "dataset_drift" : n_drifted > 0,
        "n_drifted_cols": n_drifted,
        "feature_drift" : feature_drift,
        "share_drifted" : n_drifted / len(FEATURES),
    }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

    print(f"\n[INFO] Generating drift report for {ticker} ...")
    path = generate_drift_report(ticker)
    print(f"[INFO] Open in browser: {path}")

    print(f"\n[INFO] Drift summary:")
    summary = get_drift_summary(ticker)
    print(f"  Dataset drift    : {summary['dataset_drift']}")
    print(f"  Features drifted : {summary['n_drifted_cols']} / {len(FEATURES)}")
    for feat, drifted in summary["feature_drift"].items():
        print(f"    {feat:8s} : {'DRIFT' if drifted else 'OK'}")
