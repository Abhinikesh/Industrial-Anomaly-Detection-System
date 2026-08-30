"""
Trains Isolation Forest + PyTorch Autoencoder on the Azure PdM dataset.

Features: voltage, rotation, pressure, vibration  (4 features)
Label:    failure_within_24h  (binary, from preprocess.py look-ahead labeling)

Run order:
  python ml/datasets/azure_pdm/download_dataset.py
  python ml/datasets/azure_pdm/preprocess.py
  python ml/train_azure_models.py                   ← this script

Outputs:
  models/azure/isolation_forest.pkl
  models/azure/scaler.pkl
  models/azure/autoencoder.pt
  models/azure/autoencoder_threshold.json
  results/azure_pdm/  (confusion matrices, training curve, error distribution)
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "../../data/processed/azure_pdm.csv")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "../../models/azure")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "../../results/azure_pdm")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── feature config ─────────────────────────────────────────────────────────────
FEATURES = ["voltage", "rotation", "pressure", "vibration"]
N_FEATURES = len(FEATURES)   # 4  (vs 5 for AI4I)

# Isolation Forest: contamination is computed from the actual failure rate in
# the preprocessed data (not hardcoded) so it stays correct if the lookahead
# window is changed.  'auto' would default to 10%, which vastly over-flags
# normal readings for a ~2-3% positive rate dataset.
CONTAMINATION = "measured"   # resolved dynamically in load_data()

# ── AE hyperparams ─────────────────────────────────────────────────────────────
EPOCHS         = 50
BATCH_SIZE     = 64    # larger than AI4I because we have ~10× more rows
VAL_SPLIT      = 0.20
LR             = 1e-3
THRESHOLD_SIGMA= 2     # threshold = mean + 2σ of normal reconstruction error


# ── Autoencoder architecture ───────────────────────────────────────────────────
class SensorAutoencoder(nn.Module):
    """
    4 → 3 → 2 (bottleneck) → 3 → 4

    Same bottleneck ratio as the AI4I 5-feature model.
    ReLU activations in encoder/decoder hidden layers to capture non-linear
    sensor correlations (e.g. voltage and rotation are mechanically coupled).
    Linear output — no squashing — so MSE is computed in the original real-
    valued feature space, not a compressed sigmoid range.
    """
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 3), nn.ReLU(),
            nn.Linear(3, 2),          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),          nn.ReLU(),
            nn.Linear(3, n_features),           # linear output — no activation
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── data loading ───────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Preprocessed data not found at:\n  {DATA_PATH}\n"
            "Run  python ml/datasets/azure_pdm/preprocess.py  first."
        )
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")

    # Compute contamination from actual failure rate so it adapts to
    # whatever look-ahead window was used during preprocessing.
    failure_rate = df["failure_within_24h"].mean()
    print(f"  Measured failure rate: {failure_rate * 100:.2f}%")

    # IsolationForest requires contamination in (0.0, 0.5].
    # If the rate exceeds 0.5 something is wrong with the preprocessing
    # (e.g. a labeling bug), so we cap at 0.5 and warn loudly.
    contamination = min(float(failure_rate), 0.5)
    if failure_rate > 0.5:
        print(
            f"  ⚠  WARNING: measured failure rate {failure_rate * 100:.1f}% "
            f"exceeds sklearn's 50% contamination limit. Clamping to 0.50.\n"
            f"     If this looks wrong, re-run preprocess.py to regenerate labels."
        )
    print(f"  Contamination used for IF: {contamination:.4f}")

    return df, contamination



def prepare_features(df: pd.DataFrame):
    """
    Scale the 4 sensor features and split into full array + normal-only subset.

    Why scale?
      Voltage (~150-200V) and vibration (~40-60 units) have very different
      magnitudes. Without StandardScaler, high-magnitude features dominate
      isolation tree splits and the model effectively ignores lower-magnitude
      sensors.  Mean=0, std=1 normalisation makes all four sensors equally
      influential in the anomaly score.
    """
    scaler   = StandardScaler()
    X_all    = scaler.fit_transform(df[FEATURES])
    y_all    = df["failure_within_24h"].values.astype(int)

    normal_mask = (y_all == 0)
    X_normal    = X_all[normal_mask]

    print(f"\n  Total rows           : {len(X_all):,}")
    print(f"  Normal rows (train)  : {X_normal.shape[0]:,}")
    print(f"  Failure rows (test)  : {(~normal_mask).sum():,}")

    return X_all, y_all, X_normal, scaler


# ── Isolation Forest ───────────────────────────────────────────────────────────

def train_isolation_forest(X_scaled: np.ndarray, contamination: float) -> IsolationForest:
    print(f"\nTraining Isolation Forest (contamination={contamination:.4f}, n_estimators=100)...")
    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_scaled)
    return clf


def evaluate_isolation_forest(clf: IsolationForest, X_scaled: np.ndarray, y_true: np.ndarray):
    raw_preds  = clf.predict(X_scaled)
    is_anomaly = (raw_preds == -1).astype(int)
    scores     = -clf.score_samples(X_scaled)

    prec = precision_score(y_true, is_anomaly)
    rec  = recall_score(y_true, is_anomaly)
    f1   = f1_score(y_true, is_anomaly)

    print("\n" + "=" * 60)
    print("Isolation Forest — Evaluation Results (Azure PdM)")
    print("=" * 60)
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_true, is_anomaly, target_names=["Normal", "Failure"]))

    return is_anomaly, scores, prec, rec, f1


def plot_if_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Isolation Forest — Confusion Matrix (Azure PdM)", fontsize=11, pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "if_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_if_score_distribution(scores, y_true, contamination):
    normal_scores  = scores[y_true == 0]
    failure_scores = scores[y_true == 1]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(normal_scores,  bins=80, alpha=0.65, color="#4a90d9",
            label=f"Normal  (n={len(normal_scores):,})",  density=True)
    ax.hist(failure_scores, bins=80, alpha=0.65, color="#e05252",
            label=f"Failure (n={len(failure_scores):,})", density=True)

    threshold = np.quantile(scores, 1 - contamination)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ({threshold:.3f})")
    ax.set_xlabel("Anomaly Score (higher = more anomalous)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Isolation Forest — Score Distribution (Azure PdM)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "if_score_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
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
    print(f"  AE train: {n_tr:,} samples | val: {n_val:,} samples")
    return tr_loader, val_loader


def train_autoencoder(model: SensorAutoencoder, tr_loader, val_loader) -> dict:
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"train": [], "val": []}

    print(f"\nTraining Autoencoder for {EPOCHS} epochs (4-feature Azure PdM)...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
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
    with torch.no_grad():
        t     = torch.tensor(X_np, dtype=torch.float32)
        recon = model(t).numpy()
    return np.mean((X_np - recon) ** 2, axis=1)


def compute_threshold(errors_normal: np.ndarray) -> float:
    mu     = errors_normal.mean()
    sigma  = errors_normal.std()
    thresh = mu + THRESHOLD_SIGMA * sigma
    print(f"\nAE reconstruction error on normal data:")
    print(f"  mean      = {mu:.5f}")
    print(f"  std       = {sigma:.5f}")
    print(f"  threshold (mean + {THRESHOLD_SIGMA}σ) = {thresh:.5f}")
    return float(thresh)


def evaluate_autoencoder(y_true, all_errors, threshold):
    y_pred = (all_errors > threshold).astype(int)

    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("Autoencoder — Evaluation Results (Azure PdM)")
    print("=" * 60)
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))

    return y_pred, prec, rec, f1


def plot_ae_training_curve(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="Train loss",  color="#4a90d9")
    ax.plot(history["val"],   label="Val loss",    color="#e08a52", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve (Azure PdM)")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ae_reconstruction_error(errors, y_true, threshold):
    normal_err  = errors[y_true == 0]
    failure_err = errors[y_true == 1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(normal_err,  bins=80, alpha=0.65, color="#4a90d9",
            label=f"Normal  (n={len(normal_err):,})",  density=True)
    ax.hist(failure_err, bins=80, alpha=0.65, color="#e05252",
            label=f"Failure (n={len(failure_err):,})", density=True)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.8,
               label=f"Threshold = {threshold:.4f}  (mean+{THRESHOLD_SIGMA}σ)")
    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Autoencoder — Reconstruction Error Distribution (Azure PdM)", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_reconstruction_error.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_ae_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"],
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Autoencoder — Confusion Matrix (Azure PdM)", fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "ae_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── model comparison summary ───────────────────────────────────────────────────

def print_model_comparison(if_prec, if_rec, if_f1, ae_prec, ae_rec, ae_f1):
    print("\n" + "=" * 60)
    print("Model Comparison Summary — Azure PdM Fleet Dataset")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 60)
    print(f"{'Isolation Forest':<25} {if_prec:>10.3f} {if_rec:>10.3f} {if_f1:>10.3f}")
    print(f"{'Autoencoder':<25} {ae_prec:>10.3f} {ae_rec:>10.3f} {ae_f1:>10.3f}")

    winner = "Autoencoder" if ae_f1 > if_f1 else "Isolation Forest" if if_f1 > ae_f1 else "Tied"
    print(f"\nWinner by F1: {winner}")
    print(
        "\nNote: The Azure PdM telemetry captures rotating machinery with tightly\n"
        "coupled sensor channels (voltage and rotation are mechanically linked,\n"
        "pressure and vibration correlate during load changes).  The Autoencoder\n"
        "tends to outperform Isolation Forest on datasets with strong inter-sensor\n"
        "correlations because its bottleneck captures the joint normal manifold;\n"
        "failures violate that joint structure even when individual sensor values\n"
        "look within range — exactly what reconstruction error detects."
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Azure PdM — Model Training Pipeline")
    print(f"  Features     : {FEATURES}")
    print(f"  Model output : {MODEL_DIR}/")
    print("=" * 60 + "\n")

    df, contamination = load_data()
    X_all, y_all, X_normal, scaler = prepare_features(df)

    # ── 1. Isolation Forest ──────────────────────────────────────────────────
    clf = train_isolation_forest(X_all, contamination)
    if_preds, if_scores, if_prec, if_rec, if_f1 = evaluate_isolation_forest(clf, X_all, y_all)

    print("\nGenerating Isolation Forest plots...")
    plot_if_confusion_matrix(y_all, if_preds)
    plot_if_score_distribution(if_scores, y_all, contamination)

    # ── 2. Autoencoder ───────────────────────────────────────────────────────
    tr_loader, val_loader = make_dataloaders(X_normal)
    ae_model = SensorAutoencoder(n_features=N_FEATURES)
    history  = train_autoencoder(ae_model, tr_loader, val_loader)

    all_errors    = reconstruction_errors(ae_model, X_all)
    normal_errors = reconstruction_errors(ae_model, X_normal)
    threshold     = compute_threshold(normal_errors)
    ae_preds, ae_prec, ae_rec, ae_f1 = evaluate_autoencoder(y_all, all_errors, threshold)

    print("\nGenerating Autoencoder plots...")
    plot_ae_training_curve(history)
    plot_ae_reconstruction_error(all_errors, y_all, threshold)
    plot_ae_confusion_matrix(y_all, ae_preds)

    # ── 3. Save all artefacts ────────────────────────────────────────────────
    iso_path    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    ae_path     = os.path.join(MODEL_DIR, "autoencoder.pt")
    thresh_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")

    joblib.dump(clf,    iso_path)
    joblib.dump(scaler, scaler_path)
    torch.save(ae_model.state_dict(), ae_path)

    with open(thresh_path, "w") as f:
        json.dump({
            "threshold":        threshold,
            "threshold_sigma":  THRESHOLD_SIGMA,
            "features":         FEATURES,
            "contamination":    contamination,
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Saved artefacts:")
    for path in [iso_path, scaler_path, ae_path, thresh_path]:
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path}  ({size_kb:.1f} KB)")

    print_model_comparison(if_prec, if_rec, if_f1, ae_prec, ae_rec, ae_f1)

    print(
        "\n✓  Training complete.\n"
        "Next step:\n"
        "  Restart uvicorn — the API will auto-load fleet_machine models at startup.\n"
        "  Then run:  python ml/simulator_azure.py --fast"
    )


if __name__ == "__main__":
    main()
