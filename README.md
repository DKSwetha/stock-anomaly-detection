# Stock Anomaly Detection

An end-to-end machine learning system that detects anomalies in stock market data using an LSTM Autoencoder. The system fetches live OHLCV data for any stock ticker, runs inference through a trained model, and surfaces anomalous price-volume patterns through an interactive dashboard.

---

## Architecture

```
yfinance (live data)
        |
        v
  Preprocessing          Training data (20 tickers, 2018-2024)
  (MinMaxScaler,               |
   30-day windows)             v
        |              LSTM Autoencoder (PyTorch)
        |                      |
        v                      v
   FastAPI /predict     models/lstm_autoencoder.pt
        |
        v
  Streamlit Dashboard
  (price chart, volume,
   reconstruction error,
   anomaly breakdown)
        |
        v
  Evidently AI
  (data drift monitoring)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | `yfinance`, `pandas` |
| Preprocessing | `scikit-learn` (MinMaxScaler) |
| Model | `PyTorch` (LSTM Autoencoder) |
| Experiment tracking | `MLflow` |
| API | `FastAPI`, `uvicorn` |
| Dashboard | `Streamlit`, `Plotly` |
| Monitoring | `Evidently AI` |
| Containerization | `Docker`, `docker-compose` |

---

## How it works

The core idea is reconstruction-error anomaly detection. An LSTM Autoencoder is trained on 6 years of normalized OHLCV data across 20 diverse stock tickers. It learns to compress and reconstruct "normal" 30-day price-volume sequences. At inference time, any sequence that the model reconstructs poorly — measured by mean squared error — is flagged as an anomaly.

A z-score threshold is computed fresh for each ticker: windows whose error exceeds `mean + 1.5 * std` of that ticker's own reconstruction errors are flagged. Per-feature error breakdown identifies whether the anomaly is driven by price (Open/High/Low/Close) or Volume.

---

## Project Structure

```
stock-anomaly-detection/
├── api/
│   └── main.py                 # FastAPI endpoint
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── raw/                    # Downloaded OHLCV CSVs
│   ├── processed/              # Windowed numpy arrays
│   └── scalers/                # Per-ticker MinMaxScalers
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.dashboard
├── models/
│   └── lstm_autoencoder.pt     # Trained model weights
├── monitoring/
│   └── reports/                # Evidently HTML drift reports
├── notebooks/                  # EDA and experimentation
├── src/
│   ├── data/
│   │   ├── ingest.py           # yfinance data download
│   │   └── preprocess.py       # Normalization and windowing
│   ├── model/
│   │   └── autoencoder.py      # LSTM Autoencoder architecture
│   ├── training/
│   │   └── train.py            # Training loop + MLflow tracking
│   ├── inference/
│   │   └── predict.py          # Live inference pipeline
│   └── monitoring/
│       └── monitor.py          # Evidently drift detection
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/DKSwetha/stock-anomaly-detection.git
cd stock-anomaly-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Download data and preprocess

```bash
python src/data/ingest.py
python src/data/preprocess.py
```

### 3. Train the model

```bash
python -m src.training.train
```

Track experiments at `http://127.0.0.1:5000` after running:
```bash
mlflow ui
```

### 4. Run locally

```bash
# Terminal 1 — API
uvicorn api.main:app --reload

# Terminal 2 — Dashboard
streamlit run dashboard/app.py
```

Dashboard: `http://localhost:8501`
API docs:  `http://localhost:8000/docs`

### 5. Run with Docker

```bash
docker-compose up --build
```

Dashboard: `http://localhost:8501`
API:       `http://localhost:8000`

---

## Monitoring

Generate a drift report for any ticker:

```bash
python -m src.monitoring.monitor AAPL
```

The HTML report is saved to `monitoring/reports/AAPL_drift_report.html` and compares the current market data distribution against the training distribution using Evidently AI.

---

## Model Details

| Parameter | Value |
|---|---|
| Architecture | LSTM Autoencoder |
| Input shape | (30, 5) — 30 days, 5 OHLCV features |
| Hidden size | 64 |
| Layers | 1 encoder + 1 decoder LSTM |
| Parameters | 51,781 |
| Loss function | Mean Squared Error |
| Optimizer | Adam (lr=1e-3) |
| Epochs | 30 |
| Training tickers | 20 (diversified across sectors) |
| Training period | 2018-01-01 to 2024-01-01 |

---

## Anomaly Detection Logic

1. Fetch last 90 days of OHLCV data for the selected ticker
2. Fit a fresh MinMaxScaler on this data (per-ticker normalization)
3. Create overlapping 30-day windows
4. Run each window through the trained autoencoder
5. Compute per-window MSE (reconstruction error)
6. Flag windows where `error > mean + 1.5 * std` AND `error > 0.008`
7. For flagged windows, break down error by feature to identify the primary driver

---

## Sample Output

```
Ticker  : AAPL
Windows : 32  (30-day sequences)
Threshold: 0.01741  (z-score based)
Anomalies: 3

2026-06-04 | error=0.017818 | top_feature=Volume
2026-06-05 | error=0.018871 | top_feature=Volume
2026-06-08 | error=0.018082 | top_feature=Volume
```
