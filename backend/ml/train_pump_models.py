"""
Trains Isolation Forest + PyTorch Autoencoder on the Pump Sensor dataset.

Features : top-15 sensors selected by correlation with failure label
           (sensor_04, sensor_10, sensor_11, sensor_02, sensor_12, sensor_50,
            sensor_01, sensor_03, sensor_06, sensor_07, sensor_09, sensor_08,
            sensor_05, sensor_00, sensor_40)
Label    : binary  0=NORMAL  1=BROKEN or RECOVERING

Run order:
  python ml/datasets/pump_sensor/download_dataset.py
  python ml/datasets/pump_sensor/preprocess.py
  python ml/train_pump_models.py                       ← this script

Outputs:
  models/pump/isolation_forest.pkl
  models/pump/scaler.pkl
  models/pump/autoencoder.pt
  models/pump/autoencoder_threshold.json
  results/pump_sensor/  (confusion matrices, training curve, error distribution)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, "../../data/processed/pump_sensor.csv")
MODEL_DIR   = os.path.join(SCRIPT_DIR, "../../models/pump")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "../../results/pump_sensor")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyperparams ────────────────────────────────────────────────────────────────
EPOCHS          = 60     # more epochs for 15-feature space vs 4-feature azure
BATCH_SIZE      = 128    # 220K rows — larger batch is fine
VAL_SPLIT       = 0.20
LR              = 1e-3
THRESHOLD_SIGMA = 2      # threshold = mean + 2σ of normal reconstruction error


# ── Autoencoder ────────────────────────────────────────────────────────────────
class SensorAutoencoder(nn.Module):
    """
    15 → 10 → 5 (bottleneck) → 10 → 15

    Bottleneck dimension 5 forces the encoder to learn the 5 dominant
    modes of normal pump operation (pressure cycles, flow rates, etc.)
    from the 15 correlated sensors.  Failure states distort this manifold —
    the decoder can't reconstruct them accurately → high reconstruction error.

    BatchNorm1d in the encoder stabilises training across the 15 sensors,
    which span very different physical scales (voltage, pressure, vibration).
    Dropout(0.1) prevents encoder overfitting to the large ~200K training set.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(10, 5),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, n_features),   # linear output — no activation
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── data loading ───────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Preprocessed data not found at:\n  {DATA_PATH}\n"
            "Run  python ml/datasets/pump_sensor/preprocess.py  first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")

    failure_rate = df["label"].mean()
    contamination = min(float(failure_rate), 0.5)
    print(f"  Measured failure rate: {failure_rate * 100:.2f}%  (IF contamination: {contamination:.4f})")

    # Infer feature columns: everything that starts with "sensor_"
    feature_cols = [c for c in df.columns if c.startswith("sensor_")]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    return df, feature_cols, contamination


def prepare_features(df: pd.DataFrame, feature_cols: list[str]):
    scaler  = StandardScaler()
    X_all   = scaler.fit_transform(df[feature_cols])
    y_all   = df["label"].values.astype(int)

    normal_mask = (y_all == 0)
    X_normal    = X_all[normal_mask]

    print(f"\n  Total rows           : {len(X_all):,}")
    print(f"  Normal rows (train)  : {X_normal.shape[0]:,}")
    print(f"  Failure rows (test)  : {(~normal_mask).sum():,}")

    return X_all, y_all, X_normal, scaler, feature_cols


# ── Isolation Forest ───────────────────────────────────────────────────────────

def train_isolation_forest(X_scaled: np.ndarray, contamination: float) -> IsolationForest:
    print(f"\nTraining Isolation Forest (contamination={contamination:.4f}, n_estimators=150)...")
    # 150 estimators (vs 100 in azure) — 15 features benefit from more trees
    clf = IsolationForest(n_estimators=150, contamination=contamination,
                          random_state=42, n_jobs=-1)
    clf.fit(X_scaled)
    return clf


def evaluate_if(clf, X_scaled, y_true):
    raw_preds  = clf.predict(X_scaled)
    is_anomaly = (raw_preds == -1).astype(int)
    scores     = -clf.score_samples(X_scaled)

    prec = precision_score(y_true, is_anomaly)
    rec  = recall_score(y_true, is_anomaly)
    f1   = f1_score(y_true, is_anomaly)

    print("\n" + "=" * 62)
    print("Isolation Forest — Evaluation Results (Pump Sensor)")
    print("=" * 62)
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_true, is_anomaly, target_names=["Normal", "Failure"]))
    return is_anomaly, scores, prec, rec, f1


def plot_if_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Isolation Forest — Confusion Matrix (Pump Sensor)", pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "if_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_if_scores(scores, y_true, contamination):
    normal_s  = scores[y_true == 0]
    failure_s = scores[y_true == 1]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(normal_s,  bins=80, alpha=0.65, color="#4a90d9",
            label=f"Normal  (n={len(normal_s):,})", density=True)
    ax.hist(failure_s, bins=80, alpha=0.65, color="#e05252",
            label=f"Failure (n={len(failure_s):,})", density=True)
    thresh = np.quantile(scores, 1 - contamination)
    ax.axvline(thresh, color="orange", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ({thresh:.3f})")
    ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
    ax.set_title("Isolation Forest — Score Distribution (Pump Sensor)", pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "if_score_distribution.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


# ── Autoencoder ────────────────────────────────────────────────────────────────

def make_dataloaders(X_normal: np.ndarray):
    t     = torch.tensor(X_normal, dtype=torch.float32)
    ds    = TensorDataset(t, t)
    n_val = int(len(ds) * VAL_SPLIT)
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"  AE train: {n_tr:,} | val: {n_val:,}")
    return tr_loader, val_loader


def train_autoencoder(model: SensorAutoencoder, tr_loader, val_loader) -> dict:
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train": [], "val": []}

    print(f"\nTraining Autoencoder for {EPOCHS} epochs (15-feature Pump Sensor)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_loader)

        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        if epoch % 10 == 0:
            print(f"  epoch {epoch:02d}/{EPOCHS}  "
                  f"train_loss={tr_loss:.5f}  val_loss={val_loss:.5f}")

    return history


def reconstruction_errors(model: SensorAutoencoder, X_np: np.ndarray) -> np.ndarray:
    model.eval()
    # Process in chunks to avoid OOM on 200K rows
    chunk = 8192
    errors = []
    with torch.no_grad():
        for i in range(0, len(X_np), chunk):
            xb    = torch.tensor(X_np[i:i+chunk], dtype=torch.float32)
            recon = model(xb).numpy()
            errors.append(np.mean((X_np[i:i+chunk] - recon) ** 2, axis=1))
    return np.concatenate(errors)


def compute_threshold(errors_normal: np.ndarray) -> float:
    mu     = errors_normal.mean()
    sigma  = errors_normal.std()
    thresh = mu + THRESHOLD_SIGMA * sigma
    print(f"\nAE reconstruction error (normal data):")
    print(f"  mean={mu:.5f}  std={sigma:.5f}")
    print(f"  threshold (mean+{THRESHOLD_SIGMA}σ) = {thresh:.5f}")
    return float(thresh)


def evaluate_ae(y_true, all_errors, threshold):
    y_pred = (all_errors > threshold).astype(int)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print("\n" + "=" * 62)
    print("Autoencoder — Evaluation Results (Pump Sensor)")
    print("=" * 62)
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))
    return y_pred, prec, rec, f1


def plot_ae_curve(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="Train loss",  color="#4a90d9")
    ax.plot(history["val"],   label="Val loss",    color="#e08a52", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve (Pump Sensor)")
    ax.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_training_curve.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_ae_errors(errors, y_true, threshold):
    normal_e  = errors[y_true == 0]
    failure_e = errors[y_true == 1]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(normal_e,  bins=100, alpha=0.65, color="#4a90d9",
            label=f"Normal  (n={len(normal_e):,})",  density=True)
    ax.hist(failure_e, bins=100, alpha=0.65, color="#e05252",
            label=f"Failure (n={len(failure_e):,})", density=True)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.8,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)"); ax.set_ylabel("Density")
    ax.set_title("Autoencoder — Reconstruction Error Distribution (Pump Sensor)", pad=10)
    ax.legend(fontsize=9); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_reconstruction_error.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_ae_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Autoencoder — Confusion Matrix (Pump Sensor)", pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def print_summary(feature_cols, if_prec, if_rec, if_f1, ae_prec, ae_rec, ae_f1):
    print("\n" + "=" * 62)
    print("Model Comparison Summary — Pump Sensor Dataset")
    print("=" * 62)
    print(f"\n{'Model':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    print(f"{'Isolation Forest':<22} {if_prec:>10.3f} {if_rec:>10.3f} {if_f1:>10.3f}")
    print(f"{'Autoencoder':<22} {ae_prec:>10.3f} {ae_rec:>10.3f} {ae_f1:>10.3f}")
    winner = ("Autoencoder" if ae_f1 > if_f1
              else "Isolation Forest" if if_f1 > ae_f1 else "Tied")
    print(f"\nWinner by F1: {winner}")
    print(
        "\nNote: This pump dataset has strong sequential structure — the RECOVERING\n"
        "class represents a continuous degradation trajectory.  The Autoencoder\n"
        "tends to perform better here because the bottleneck latent space captures\n"
        "the multi-sensor 'normal operating manifold'; RECOVERING states deviate\n"
        "from this manifold in complex, correlated ways that tree-based IF misses."
    )
    print(f"\nFeatures used: {feature_cols}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("Pump Sensor — Model Training Pipeline")
    print(f"  Model output : {MODEL_DIR}/")
    print("=" * 62 + "\n")

    df, feature_cols, contamination = load_data()
    n_features = len(feature_cols)
    X_all, y_all, X_normal, scaler, feature_cols = prepare_features(df, feature_cols)

    # ── 1. Isolation Forest ──────────────────────────────────────────────────
    clf = train_isolation_forest(X_all, contamination)
    if_preds, if_scores, if_prec, if_rec, if_f1 = evaluate_if(clf, X_all, y_all)

    print("\nGenerating Isolation Forest plots...")
    plot_if_confusion(y_all, if_preds)
    plot_if_scores(if_scores, y_all, contamination)

    # ── 2. Autoencoder ───────────────────────────────────────────────────────
    tr_loader, val_loader = make_dataloaders(X_normal)
    ae_model  = SensorAutoencoder(n_features=n_features)
    history   = train_autoencoder(ae_model, tr_loader, val_loader)

    all_errors    = reconstruction_errors(ae_model, X_all)
    normal_errors = reconstruction_errors(ae_model, X_normal)
    threshold     = compute_threshold(normal_errors)
    ae_preds, ae_prec, ae_rec, ae_f1 = evaluate_ae(y_all, all_errors, threshold)

    print("\nGenerating Autoencoder plots...")
    plot_ae_curve(history)
    plot_ae_errors(all_errors, y_all, threshold)
    plot_ae_confusion(y_all, ae_preds)

    # ── 3. Save artefacts ─────────────────────────────────────────────────────
    iso_path    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    ae_path     = os.path.join(MODEL_DIR, "autoencoder.pt")
    thresh_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")

    joblib.dump(clf,    iso_path)
    joblib.dump(scaler, scaler_path)
    torch.save(ae_model.state_dict(), ae_path)

    with open(thresh_path, "w") as f:
        json.dump({
            "threshold":       threshold,
            "threshold_sigma": THRESHOLD_SIGMA,
            "features":        feature_cols,
            "contamination":   contamination,
        }, f, indent=2)

    print("\n" + "=" * 62)
    print("Saved artefacts:")
    for path in [iso_path, scaler_path, ae_path, thresh_path]:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {os.path.basename(path):<40}  ({size_kb:.1f} KB)")

    print_summary(feature_cols, if_prec, if_rec, if_f1, ae_prec, ae_rec, ae_f1)

    print(
        "\n✓  Training complete.\n"
        "Next step:\n"
        "  Restart uvicorn — water_pump models will load at startup.\n"
        "  Then run all 3 simulators:\n"
        "    python ml/simulator.py --fast\n"
        "    python ml/simulator_azure.py --fast\n"
        "    python ml/simulator_pump.py --fast"
    )


if __name__ == "__main__":
    main()
