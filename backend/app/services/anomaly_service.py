"""
Loads both trained models once at module import time so each /ingest
request doesn't re-read files from disk.

We load: Isolation Forest (joblib), StandardScaler (joblib),
         Autoencoder weights (PyTorch), and the threshold JSON.
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "../../../models")

FEATURE_ORDER = [
    "air_temp",
    "process_temp",
    "rpm",
    "torque",
    "tool_wear",
]


# ── Autoencoder definition (must match train_autoencoder.py) ───────────────────

class SensorAutoencoder(nn.Module):
    def __init__(self, n_features=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 3), nn.ReLU(),
            nn.Linear(3, 2),          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),          nn.ReLU(),
            nn.Linear(3, n_features),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── load once ──────────────────────────────────────────────────────────────────

def _load_models():
    iso_path    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    ae_path     = os.path.join(MODEL_DIR, "autoencoder.pt")
    thresh_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")

    missing = [p for p in [iso_path, scaler_path, ae_path, thresh_path]
               if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing model files — run the training scripts first:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    iso    = joblib.load(iso_path)
    scaler = joblib.load(scaler_path)

    ae = SensorAutoencoder(n_features=5)
    ae.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
    ae.eval()

    with open(thresh_path) as f:
        thresh_cfg = json.load(f)
    ae_threshold = thresh_cfg["threshold"]

    return iso, scaler, ae, ae_threshold


try:
    _iso, _scaler, _ae, _ae_threshold = _load_models()
    _models_ready = True
except FileNotFoundError as e:
    print(f"[anomaly_service] WARNING: {e}")
    _models_ready = False


# ── scoring ────────────────────────────────────────────────────────────────────

def score_reading(payload: dict) -> dict:
    """
    Takes a dict with the 5 sensor values, returns scores + flags from both models.
    Falls back to placeholder zeros if models aren't loaded yet.
    """
    if not _models_ready:
        return {
            "iso_score": 0.0, "iso_flag": 0,
            "ae_score":  0.0, "ae_flag":  0,
            "is_anomaly": False,
        }

    # extract in the same order the scaler was fit on
    raw = np.array([[
        payload["air_temp"],
        payload["process_temp"],
        payload["rpm"],
        payload["torque"],
        payload["tool_wear"],
    ]])

    X = _scaler.transform(raw)

    # Isolation Forest: score_samples returns negative path length;
    # we negate so higher = more anomalous; predict() → -1 means anomaly
    iso_raw_score = float(-_iso.score_samples(X)[0])
    iso_pred      = _iso.predict(X)[0]       # -1 or +1
    iso_flag      = 1 if iso_pred == -1 else 0

    # Autoencoder: row-wise MSE between original and reconstruction
    t      = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = _ae(t).numpy()
    ae_score = float(np.mean((X - recon) ** 2))
    ae_flag  = 1 if ae_score > _ae_threshold else 0

    # OR logic: flag as anomaly if *either* model says so.
    # This maximises recall — we'd rather check a false alarm than miss a
    # real failure.  The trade-off is slightly more noise in the alert feed,
    # which is acceptable in an industrial safety context.
    is_anomaly = bool(iso_flag or ae_flag)

    return {
        "iso_score":  round(iso_raw_score, 5),
        "iso_flag":   iso_flag,
        "ae_score":   round(ae_score, 5),
        "ae_flag":    ae_flag,
        "is_anomaly": is_anomaly,
    }
