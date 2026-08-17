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


# ── load once with explicit error reporting ────────────────────────────────────

_iso = None
_scaler = None
_ae = None
_ae_threshold = None
_models_ready = False
_model_error_msg = ""


def load_models_safely():
    global _iso, _scaler, _ae, _ae_threshold, _models_ready, _model_error_msg

    iso_path    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    ae_path     = os.path.join(MODEL_DIR, "autoencoder.pt")
    thresh_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")

    missing = []
    if not os.path.exists(iso_path):
        missing.append(f"Isolation Forest: {iso_path}")
    if not os.path.exists(scaler_path):
        missing.append(f"Scaler: {scaler_path}")
    if not os.path.exists(ae_path):
        missing.append(f"Autoencoder Weights: {ae_path}")
    if not os.path.exists(thresh_path):
        missing.append(f"Autoencoder Threshold: {thresh_path}")

    if missing:
        _models_ready = False
        _model_error_msg = "Missing model files. Run `python ml/train_isolation_forest.py` and `python ml/train_autoencoder.py`."
        print("\n" + "=" * 70)
        print("⚠️  [ANOMALY SERVICE WARNING] Model files not found in models/ directory!")
        print("Missing required artifacts:")
        for m in missing:
            print(f"  - {m}")
        print("\nTo generate and save all model artifacts, run:")
        print("  cd backend")
        print("  python ml/train_isolation_forest.py")
        print("  python ml/train_autoencoder.py")
        print("=" * 70 + "\n")
        return False

    try:
        _iso    = joblib.load(iso_path)
        _scaler = joblib.load(scaler_path)

        ae_model = SensorAutoencoder(n_features=5)
        ae_model.load_state_dict(torch.load(ae_path, map_location="cpu", weights_only=True))
        ae_model.eval()
        _ae = ae_model

        with open(thresh_path) as f:
            thresh_cfg = json.load(f)
        _ae_threshold = thresh_cfg["threshold"]

        _models_ready = True
        _model_error_msg = ""
        print("[anomaly_service] ✓ All models (Isolation Forest + Autoencoder) successfully loaded.")
        return True
    except Exception as err:
        _models_ready = False
        _model_error_msg = f"Error loading models: {err}"
        print(f"[anomaly_service] ⚠️ Error loading models: {err}")
        return False


# Initial load at module import
load_models_safely()


# ── scoring ────────────────────────────────────────────────────────────────────

def score_reading(payload: dict) -> dict:
    """
    Takes a dict with the 5 sensor values, returns scores + flags from both models.
    If models aren't ready, returns a fallback dictionary with model_warning.
    """
    if not _models_ready:
        return {
            "iso_score": 0.0,
            "iso_flag": 0,
            "ae_score":  0.0,
            "ae_flag":   0,
            "is_anomaly": False,
            "model_warning": _model_error_msg or "Models not loaded. Train models before scoring.",
        }

    try:
        raw = np.array([[
            payload["air_temp"],
            payload["process_temp"],
            payload["rpm"],
            payload["torque"],
            payload["tool_wear"],
        ]])

        X = _scaler.transform(raw)

        # Isolation Forest scoring
        iso_raw_score = float(-_iso.score_samples(X)[0])
        iso_pred      = _iso.predict(X)[0]       # -1 or +1
        iso_flag      = 1 if iso_pred == -1 else 0

        # Autoencoder scoring
        t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = _ae(t).numpy()
        ae_score = float(np.mean((X - recon) ** 2))
        ae_flag  = 1 if ae_score > _ae_threshold else 0

        # Union ensemble logic (OR)
        is_anomaly = bool(iso_flag or ae_flag)

        return {
            "iso_score":  round(iso_raw_score, 5),
            "iso_flag":   iso_flag,
            "ae_score":   round(ae_score, 5),
            "ae_flag":    ae_flag,
            "is_anomaly": is_anomaly,
        }
    except Exception as e:
        print(f"[anomaly_service] Scoring exception: {e}")
        return {
            "iso_score": 0.0,
            "iso_flag": 0,
            "ae_score": 0.0,
            "ae_flag": 0,
            "is_anomaly": False,
            "model_warning": f"Scoring failure: {e}",
        }
