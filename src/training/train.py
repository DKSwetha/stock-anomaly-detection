"""
src/training/train.py
----------------------
Trains the LSTM Autoencoder on the combined multi-ticker dataset,
tracks experiments with MLflow, and saves the trained model weights.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch

# Allow importing sibling packages (src/data, src/model) when run directly
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.preprocess import load_processed
from src.data.ingest import TICKERS
from src.model.autoencoder import LSTMAutoencoder


# ── Configuration ──────────────────────────────────────────────────────────────
WINDOW_SIZE  = 30
N_FEATURES   = 5
HIDDEN_SIZE  = 64
NUM_LAYERS   = 1

BATCH_SIZE   = 64
EPOCHS       = 30
LEARNING_RATE = 1e-3

MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "lstm_autoencoder.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ───────────────────────────────────────────────────────────────────────────────
def get_dataloaders(batch_size: int = BATCH_SIZE):
    """
    Load the combined preprocessed windows and wrap them in DataLoaders.

    Returns:
        (train_loader, test_loader)
    """
    X_train, X_test = load_processed(name="combined")

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion):
    """Run one training epoch. Returns average training loss."""
    model.train()
    total_loss = 0.0

    for (batch,) in loader:
        batch = batch.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    """Run evaluation (no gradient updates). Returns average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            total_loss += loss.item() * batch.size(0)

    return total_loss / len(loader.dataset)


def train_model():
    """
    Full training pipeline:
    load data -> build model -> train loop with MLflow logging -> save model
    """
    print(f"[INFO] Using device: {DEVICE}")

    train_loader, test_loader = get_dataloaders()
    print(f"[INFO] Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    model = LSTMAutoencoder(
        n_features=N_FEATURES,
        window_size=WINDOW_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    mlflow.set_experiment("stock-anomaly-detection")

    with mlflow.start_run(run_name="lstm-autoencoder-multistock"):
        # Log hyperparameters and dataset info
        mlflow.log_params({
            "window_size": WINDOW_SIZE,
            "n_features": N_FEATURES,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "n_tickers": len(TICKERS),
            "tickers": ",".join(TICKERS),
        })

        best_val_loss = float("inf")

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss = evaluate(model, test_loader, criterion)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

            print(f"[Epoch {epoch:2d}/{EPOCHS}] "
                  f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), MODEL_PATH)

        mlflow.log_metric("best_val_loss", best_val_loss)

        # Log the final/best model as an MLflow artifact too
        model.load_state_dict(torch.load(MODEL_PATH))
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"\n[INFO] Training complete. Best val_loss = {best_val_loss:.6f}")
        print(f"[INFO] Best model saved -> {MODEL_PATH}")

    return model, best_val_loss


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, best_val_loss = train_model()

