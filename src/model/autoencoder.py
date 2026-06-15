"""
src/model/autoencoder.py
-------------------------
LSTM Autoencoder for anomaly detection on multivariate
time-series windows of shape (window_size, n_features).

The model learns to compress a sequence into a small latent
vector (encoder) and reconstruct the original sequence from it
(decoder). Windows that the model reconstructs poorly (high
reconstruction error) are flagged as anomalies.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    Encoder-decoder LSTM autoencoder.

    Args:
        n_features  : Number of features per time step (5 for OHLCV)
        window_size : Number of time steps per window (30)
        hidden_size : Size of the LSTM hidden state / latent vector
        num_layers  : Number of stacked LSTM layers in encoder/decoder
    """

    def __init__(self, n_features: int = 5, window_size: int = 30,
                 hidden_size: int = 64, num_layers: int = 1):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ── Encoder ──────────────────────────────────────────────────
        # Reads the input sequence and compresses it into a final
        # hidden state (the "latent representation" of the window).
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # ── Decoder ──────────────────────────────────────────────────
        # Takes the latent representation, repeated for every time
        # step, and reconstructs the sequence step by step.
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Projects each decoder hidden state back to n_features
        # (i.e. back to Open/High/Low/Close/Volume scale).
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, window_size, n_features)

        Returns:
            Reconstructed tensor of shape (batch_size, window_size, n_features)
        """
        batch_size = x.size(0)

        # --- Encode ---
        # _ , (h_n, c_n): h_n is the final hidden state of the encoder,
        # shape (num_layers, batch_size, hidden_size). This is the
        # "summary" of the whole input window.
        _, (h_n, _) = self.encoder(x)

        # Take the last layer's hidden state as the latent vector.
        latent = h_n[-1]  # shape: (batch_size, hidden_size)

        # --- Repeat latent vector across time steps ---
        # The decoder needs an input at every time step, so we
        # repeat the same latent vector window_size times.
        decoder_input = latent.unsqueeze(1).repeat(1, self.window_size, 1)
        # shape: (batch_size, window_size, hidden_size)

        # --- Decode ---
        decoder_output, _ = self.decoder(decoder_input)
        # shape: (batch_size, window_size, hidden_size)

        # --- Project back to original feature space ---
        reconstruction = self.output_layer(decoder_output)
        # shape: (batch_size, window_size, n_features)

        return reconstruction


# ── Quick smoke-test when run directly ──────────────────────────────────────────
if __name__ == "__main__":
    model = LSTMAutoencoder(n_features=5, window_size=30, hidden_size=64)

    # Fake batch: 8 windows, each 30 time steps, 5 features
    dummy_input = torch.randn(8, 30, 5)
    output = model(dummy_input)

    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape must match input shape!"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {n_params:,}")
    print("\nModel architecture:")
    print(model)
