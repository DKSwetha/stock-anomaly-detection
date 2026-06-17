"""
api/main.py
-----------
FastAPI application exposing an anomaly-detection endpoint.

Endpoints:
  GET  /                  -> health check
  GET  /predict/{ticker}  -> run anomaly analysis for a ticker
"""

import os
import sys
from fastapi import Query
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.inference.predict import analyze_ticker


app = FastAPI(
    title="Stock Anomaly Detection API",
    description="Detects anomalies in stock price/volume sequences using an LSTM Autoencoder.",
    version="1.0.0",
)

# Allow the Streamlit dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "Stock Anomaly Detection API is running"}


@app.get("/predict/{ticker}")
def predict(ticker: str, lookback_days: int = Query(default=90)):
    """
    Run anomaly detection for the given ticker.

    Args:
        ticker: Stock symbol, e.g. 'AAPL', 'TSLA'

    Returns:
        JSON with dates, prices, reconstruction errors, anomaly flags,
        and per-feature breakdown for anomalous windows.
    """
    ticker = ticker.upper().strip()

    try:
        result = analyze_ticker(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    return result


# ── Run with: uvicorn api.main:app --reload ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
