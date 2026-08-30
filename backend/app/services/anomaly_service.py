"""
Anomaly scoring engine with a machine-type model registry.

Models are loaded once at module import time (not per-request) for speed.

Architecture
------------
_MODEL_REGISTRY maps machine_type -> bundle dict containing:
    iso          : trained IsolationForest
    scaler       : fitted StandardScaler (same feature space as IF)
    ae           : loaded SensorAutoencoder (eval mode)
    ae_threshold : float — reconstruction error cutoff (mean + 2σ)
    feature_order: list[str] — key order into sensor_values dict

Adding a new machine type in the future:
  1. Train & save new model artefacts.
  2. Add a new entry to _MACHINE_MODEL_PATHS.
  3. Everything else (scoring, routing, storage) works automatically.
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "../../../models")


# ── Autoencoder definition (must match train_autoencoder.py) ──────────────────

class SensorAutoencoder(nn.Module):
    def __init__(self, n_features: int = 5):
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


# ── Machine-type → model file paths ──────────────────────────────────────────
# To register a new machine type, add a new entry here with paths to its
# trained artefacts and the feature_order that matches its scaler/model.

_MACHINE_MODEL_PATHS: Dict[str, dict] = {
    "milling_machine": {
        "iso_path":    os.path.join(MODEL_DIR, "isolation_forest.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "scaler.pkl"),
        "ae_path":     os.path.join(MODEL_DIR, "autoencoder.pt"),
        "thresh_path": os.path.join(MODEL_DIR, "autoencoder_threshold.json"),
        # must match the column order used during scaler/model training
        "feature_order": [
            "air_temp",
            "process_temp",
            "rpm",
            "torque",
            "tool_wear",
        ],
        "n_features": 5,
    },
    # Azure Predictive Maintenance — 100-machine fleet (4-sensor telemetry)
    # Trained by:  python ml/train_azure_models.py
    # Dataset:     microsoft-azure-predictive-maintenance (Kaggle)
    # Features:    voltage, rotation, pressure, vibration
    "fleet_machine": {
        "iso_path":    os.path.join(MODEL_DIR, "azure/isolation_forest.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "azure/scaler.pkl"),
        "ae_path":     os.path.join(MODEL_DIR, "azure/autoencoder.pt"),
        "thresh_path": os.path.join(MODEL_DIR, "azure/autoencoder_threshold.json"),
        "feature_order": ["voltage", "rotation", "pressure", "vibration"],
        "n_features": 4,
    },
    # Pump Sensor Data — single industrial water pump (15 sensors selected by correlation)
    # Trained by:  python ml/train_pump_models.py
    # Dataset:     nphantawee/pump-sensor-data (Kaggle)
    # Features:    top-15 sensors from sensor_00…sensor_51 (see autoencoder_threshold.json)
    # Note: feature_order is populated at load time from the saved threshold JSON so that
    #       the exact sensor list is always in sync with the trained model artefacts.
    "water_pump": {
        "iso_path":    os.path.join(MODEL_DIR, "pump/isolation_forest.pkl"),
        "scaler_path": os.path.join(MODEL_DIR, "pump/scaler.pkl"),
        "ae_path":     os.path.join(MODEL_DIR, "pump/autoencoder.pt"),
        "thresh_path": os.path.join(MODEL_DIR, "pump/autoencoder_threshold.json"),
        # feature_order is loaded from thresh_path at startup (set to None here
        # so load_models_safely knows to read it from the JSON file)
        "feature_order": None,
        "n_features": None,   # resolved from scaler at load time
    },
    # Add future machine types here following the same pattern.
    # Each type needs its own trained artefacts in a subdirectory of models/.
}



# ── Runtime model registry (populated by load_models_safely) ─────────────────

_MODEL_REGISTRY:   Dict[str, dict] = {}   # machine_type -> loaded bundle
_registry_errors:  Dict[str, str]  = {}   # machine_type -> error msg if any


def load_models_safely() -> None:
    """Load all registered machine-type models into _MODEL_REGISTRY.

    Called once at module import.  Failures per machine type are logged but
    don't prevent other types from loading.
    """
    for machine_type, paths in _MACHINE_MODEL_PATHS.items():
        _load_single_type(machine_type, paths)


def _load_single_type(machine_type: str, paths: dict) -> None:
    iso_path    = paths["iso_path"]
    scaler_path = paths["scaler_path"]
    ae_path     = paths["ae_path"]
    thresh_path = paths["thresh_path"]

    missing = []
    for label, p in [
        ("Isolation Forest",  iso_path),
        ("Scaler",            scaler_path),
        ("Autoencoder Weights", ae_path),
        ("Autoencoder Threshold", thresh_path),
    ]:
        if not os.path.exists(p):
            missing.append(f"{label}: {p}")

    if missing:
        msg = (
            f"Missing model files for machine_type='{machine_type}'. "
            "Run the training scripts first.\n  " + "\n  ".join(missing)
        )
        _registry_errors[machine_type] = msg
        print(f"\n[anomaly_service] ⚠️  {msg}\n")
        return

    try:
        iso    = joblib.load(iso_path)
        scaler = joblib.load(scaler_path)

        # Load the threshold JSON first — it also stores feature_order for
        # types (like water_pump) that don't hardcode it in _MACHINE_MODEL_PATHS.
        with open(thresh_path) as f:
            thresh_data  = json.load(f)
        ae_threshold  = thresh_data["threshold"]

        # Resolve feature_order: prefer the hardcoded list from _MACHINE_MODEL_PATHS
        # (guaranteed correct), fall back to the list saved in the threshold JSON.
        feature_order = paths.get("feature_order") or thresh_data.get("features")

        # Resolve n_features: prefer hardcoded, else derive from scaler.
        n_features = paths.get("n_features") or (
            scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else len(feature_order)
        )

        ae_model = SensorAutoencoder(n_features=n_features)
        ae_model.load_state_dict(
            torch.load(ae_path, map_location="cpu", weights_only=True)
        )
        ae_model.eval()

        _MODEL_REGISTRY[machine_type] = {
            "iso":          iso,
            "scaler":       scaler,
            "ae":           ae_model,
            "ae_threshold": ae_threshold,
            "feature_order": feature_order,
        }
        print(
            f"[anomaly_service] ✓ Models loaded for machine_type='{machine_type}' "
            f"(IF + AE, {n_features} features, threshold={ae_threshold:.5f})"
        )
    except Exception as err:
        msg = f"Error loading models for '{machine_type}': {err}"
        _registry_errors[machine_type] = msg
        print(f"[anomaly_service] ⚠️  {msg}")



# Initial load at module import
load_models_safely()


# ── Per-request scoring ───────────────────────────────────────────────────────

def score_reading(payload: dict) -> dict:
    """
    Score one sensor reading using the model bundle for its machine_type.

    payload must contain:
      - machine_type : str  (default "milling_machine")
      - sensor_values: dict  (keys must match the bundle's feature_order)

    Returns a dict with iso_score, iso_flag, ae_score, ae_flag, is_anomaly.
    On failure, returns zeroed-out fallback with a model_warning key.
    """
    machine_type  = payload.get("machine_type", "milling_machine")
    sensor_values = payload.get("sensor_values", {})

    # ── Look up the correct model bundle ─────────────────────────────────────
    bundle = _MODEL_REGISTRY.get(machine_type)
    if bundle is None:
        err_msg = _registry_errors.get(
            machine_type,
            f"No models registered for machine_type='{machine_type}'. "
            "Add artefact paths to _MACHINE_MODEL_PATHS and retrain.",
        )
        return _fallback(err_msg)

    # ── Build feature vector in correct order ─────────────────────────────────
    feature_order = bundle["feature_order"]
    try:
        raw_values = [float(sensor_values[k]) for k in feature_order]
    except KeyError as ke:
        return _fallback(
            f"sensor_values missing key {ke} required for '{machine_type}'. "
            f"Expected keys: {feature_order}"
        )

    # ── Score with Isolation Forest ───────────────────────────────────────────
    try:
        scaler = bundle["scaler"]
        iso    = bundle["iso"]
        ae     = bundle["ae"]
        thresh = bundle["ae_threshold"]

        X = scaler.transform([raw_values])

        iso_raw_score = float(-iso.score_samples(X)[0])
        iso_pred      = iso.predict(X)[0]          # -1 = anomaly, +1 = normal
        iso_flag      = 1 if iso_pred == -1 else 0

        # ── Score with Autoencoder ────────────────────────────────────────────
        t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            recon = ae(t).numpy()
        ae_score = float(np.mean((X - recon) ** 2))
        ae_flag  = 1 if ae_score > thresh else 0

        # ── Union ensemble (OR) ───────────────────────────────────────────────
        is_anomaly = bool(iso_flag or ae_flag)

        return {
            "iso_score":  round(iso_raw_score, 5),
            "iso_flag":   iso_flag,
            "ae_score":   round(ae_score, 5),
            "ae_flag":    ae_flag,
            "is_anomaly": is_anomaly,
        }

    except Exception as e:
        print(f"[anomaly_service] Scoring exception for '{machine_type}': {e}")
        return _fallback(f"Scoring failure for '{machine_type}': {e}")


def _fallback(warning: str) -> dict:
    return {
        "iso_score":     0.0,
        "iso_flag":      0,
        "ae_score":      0.0,
        "ae_flag":       0,
        "is_anomaly":    False,
        "model_warning": warning,
    }
