"""
Trains Isolation Forest and a simple Autoencoder on simulated sensor data.
Run this once before starting the API so the models are saved to ../../models/.
"""

import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from simulator import stream

# ── config ────────────────────────────────────────────────────────────────────
N_SAMPLES   = 2000
CONTAM_RATE = 0.05
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "../../models")
FEATURES    = ["temperature", "vibration", "pressure", "torque", "tool_wear"]
EPOCHS      = 30
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR, exist_ok=True)


class SensorAutoencoder(nn.Module):
    """Tiny autoencoder — 5 inputs → bottleneck of 4 → back to 5."""
    def __init__(self, input_dim=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8), nn.ReLU(),
            nn.Linear(8, 4),         nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),          nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def generate_training_data(n=N_SAMPLES):
    """Pull n readings from the simulator (no time delay)."""
    gen = stream(anomaly_prob=CONTAM_RATE, delay=0)
    rows = [next(gen) for _ in range(n)]
    return pd.DataFrame(rows)[FEATURES]


def train_isolation_forest(X_scaled):
    clf = IsolationForest(contamination=CONTAM_RATE, random_state=42, n_estimators=100)
    clf.fit(X_scaled)
    return clf


def train_autoencoder(X_normal_np):
    X_t = torch.tensor(X_normal_np, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_t, X_t), batch_size=32, shuffle=True)

    model = SensorAutoencoder(input_dim=X_normal_np.shape[1])
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/len(loader):.4f}")

    return model


def main():
    print("Generating training data...")
    df = generate_training_data()

    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    print("Training Isolation Forest...")
    iso = train_isolation_forest(X)

    # filter to roughly-normal samples before training the autoencoder
    iso_labels = iso.predict(X)
    X_normal = X[iso_labels == 1]

    print("Training Autoencoder (PyTorch)...")
    ae = train_autoencoder(X_normal)

    # save everything
    with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    torch.save(ae.state_dict(), os.path.join(MODEL_DIR, "autoencoder.pt"))

    print(f"Models saved to {MODEL_DIR}")

    # TODO: add cross-validation and proper train/val split
    # TODO: log metrics (precision, recall) so we can track model drift over time
    # TODO: support retraining on live data collected from MongoDB


if __name__ == "__main__":
    main()
