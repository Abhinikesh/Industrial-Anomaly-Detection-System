"""
Trains a reconstruction-based Autoencoder on the AI4I 2020 sensor data.

The core idea:
  - Train ONLY on normal rows  →  the model learns what "normal" looks like
  - At inference, pass any reading through  →  normal rows reconstruct well,
    anomalous rows have noticeably higher reconstruction error (MSE)
  - Set a threshold on that error distribution to decide "anomaly or not"

Run order: download_dataset.py → explore_data.py → train_isolation_forest.py → this script
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "../../data/raw/ai4i2020.csv")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "../../models")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "../../results/autoencoder")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── hyperparams ────────────────────────────────────────────────────────────────
FEATURES    = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
EPOCHS      = 50
BATCH_SIZE  = 32
VAL_SPLIT   = 0.20
LR          = 1e-3
# threshold: flag anything above (mean + N * std) of normal reconstruction error.
# 2σ ≈ 97.7th percentile of normal error — catches rare but clear anomalies.
# 3σ is more conservative (fewer false alarms, possibly more missed failures).
# We use 2σ here to favour recall over precision, which matters more for
# industrial safety — better to investigate a false alarm than miss a failure.
THRESHOLD_SIGMA = 2


# ── model ──────────────────────────────────────────────────────────────────────

class SensorAutoencoder(nn.Module):
    """
    5 → 3 → 2 (bottleneck) → 3 → 5

    ReLU on hidden layers so the encoder can learn non-linear relationships
    between sensors (e.g. torque↔RPM anti-correlation).
    Linear output so reconstructed values aren't squashed — we need the
    full real-valued range to compute a meaningful MSE.
    """
    def __init__(self, n_features=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 3), nn.ReLU(),
            nn.Linear(3, 2),          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),          nn.ReLU(),
            nn.Linear(3, n_features),           # linear — no activation
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── data loading ───────────────────────────────────────────────────────────────

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run download_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")
    return df


def prepare(df):
    # Re-fit scaler on the full dataset for consistent scaling with the
    # Isolation Forest scaler. In practice you'd load the saved scaler.pkl,
    # but refitting here keeps this script self-contained.
    scaler  = StandardScaler()
    X_all   = scaler.fit_transform(df[FEATURES])
    y_all   = df["Machine failure"].values

    # Training: ONLY normal rows — autoencoder must not see failures during
    # training, otherwise it learns to reconstruct them too and loses its
    # ability to detect them as unusual.
    normal_mask = (y_all == 0)
    X_normal    = X_all[normal_mask]

    print(f"  Normal rows (train pool): {X_normal.shape[0]:,}")
    print(f"  Failure rows (test only): {(~normal_mask).sum():,}")

    return X_all, y_all, X_normal, scaler


# ── training ───────────────────────────────────────────────────────────────────

def make_dataloaders(X_normal):
    t      = torch.tensor(X_normal, dtype=torch.float32)
    ds     = TensorDataset(t, t)
    n_val  = int(len(ds) * VAL_SPLIT)
    n_tr   = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val],
                                 generator=torch.Generator().manual_seed(42))
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print(f"  Train: {n_tr} samples  |  Val: {n_val} samples")
    return tr_loader, val_loader


def train(model, tr_loader, val_loader):
    # MSE loss: measures how far the reconstructed vector is from the original.
    # For anomaly detection this is perfect — a high MSE means the model
    # couldn't reproduce the reading, which signals "unusual input."
    loss_fn = nn.MSELoss()
    opt     = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train": [], "val": []}

    print(f"\nTraining for {EPOCHS} epochs...")
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


# ── evaluation ─────────────────────────────────────────────────────────────────

def reconstruction_errors(model, X_np):
    """Row-wise MSE between original and reconstructed sensor values."""
    model.eval()
    with torch.no_grad():
        t      = torch.tensor(X_np, dtype=torch.float32)
        recon  = model(t).numpy()
    # mean across the 5 features for each row
    errors = np.mean((X_np - recon) ** 2, axis=1)
    return errors


def compute_threshold(errors_normal):
    """
    threshold = mean + THRESHOLD_SIGMA * std of reconstruction errors
    computed on normal-only training data.

    This is the standard statistical approach: we assume normal errors follow
    roughly a Gaussian distribution, so mean+2σ catches ~97.7% of normal
    variation.  Anything above is treated as an anomaly.
    """
    mu    = errors_normal.mean()
    sigma = errors_normal.std()
    thresh = mu + THRESHOLD_SIGMA * sigma
    print(f"\nReconstruction error on normal data:")
    print(f"  mean  = {mu:.5f}")
    print(f"  std   = {sigma:.5f}")
    print(f"  threshold (mean + {THRESHOLD_SIGMA}σ) = {thresh:.5f}")
    return thresh


def evaluate(y_true, y_pred, model_name="Autoencoder"):
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"\n{'=' * 58}")
    print(f"{model_name} — Evaluation Results")
    print(f"{'=' * 58}")
    print(f"\n  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")
    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))
    return prec, rec, f1


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_training_curve(history):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="Train loss",  color="#4a90d9")
    ax.plot(history["val"],   label="Val loss",    color="#e08a52", linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Autoencoder Training Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "training_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_reconstruction_error(errors, y_true, threshold):
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
    ax.set_title("Autoencoder — Reconstruction Error Distribution", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "reconstruction_error.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Failure"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Autoencoder — Confusion Matrix", fontsize=12, pad=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── comparison ─────────────────────────────────────────────────────────────────

def compare_with_isolation_forest(ae_prec, ae_rec, ae_f1):
    # Isolation Forest results from train_isolation_forest.py
    if_prec, if_rec, if_f1 = 0.168, 0.168, 0.168

    print("\n" + "=" * 58)
    print("Model Comparison")
    print("=" * 58)
    print(f"\n{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 58)
    print(f"{'Isolation Forest':<25} {if_prec:>10.3f} {if_rec:>10.3f} {if_f1:>10.3f}")
    print(f"{'Autoencoder':<25} {ae_prec:>10.3f} {ae_rec:>10.3f} {ae_f1:>10.3f}")

    if ae_f1 > if_f1:
        winner = "Autoencoder"
        reason = (
            "The Autoencoder outperforms Isolation Forest here likely because "
            "failures in this dataset aren't isolated point outliers — they're "
            "readings that violate learned *relationships* between sensors "
            "(e.g. torque and RPM move together in a specific way during normal "
            "operation). The Autoencoder's bottleneck forces it to learn that "
            "joint structure, so it reconstructs unusual combinations poorly. "
            "Isolation Forest only looks at individual feature distributions, "
            "missing these cross-sensor patterns."
        )
    elif if_f1 > ae_f1:
        winner = "Isolation Forest"
        reason = (
            "Isolation Forest edges out the Autoencoder here — some failure "
            "readings are genuine outliers in individual sensor dimensions "
            "(e.g. very high torque), which tree-based isolation detects "
            "directly. The Autoencoder's bottleneck may be *too* small (2 dims) "
            "for 5 features, causing it to blur the reconstruction boundary. "
            "Increasing the bottleneck or training epochs could close the gap."
        )
    else:
        winner = "Both tied"
        reason = (
            "Both models achieve similar F1 on this dataset. In practice "
            "combining their scores (ensemble) would likely improve results."
        )

    print(f"\nWinner: {winner}")
    print(f"\n{reason}\n")

    # TODO: try an ensemble score (average of IF and AE anomaly scores)
    #       before committing to one model for the API


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    df = load_data()
    X_all, y_all, X_normal, scaler = prepare(df)

    tr_loader, val_loader = make_dataloaders(X_normal)

    model = SensorAutoencoder(n_features=len(FEATURES))
    history = train(model, tr_loader, val_loader)

    # compute threshold from reconstruction errors on normal training data only
    all_errors     = reconstruction_errors(model, X_all)
    normal_errors  = reconstruction_errors(model, X_normal)
    threshold      = compute_threshold(normal_errors)

    # flag anomaly: 1 if error > threshold, else 0
    y_pred = (all_errors > threshold).astype(int)

    ae_prec, ae_rec, ae_f1 = evaluate(y_all, y_pred)

    print("\nGenerating plots...")
    plot_training_curve(history)
    plot_reconstruction_error(all_errors, y_all, threshold)
    plot_confusion_matrix(y_all, y_pred)

    # save model
    model_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")

    # save threshold so the API can load it without recomputing
    thresh_path = os.path.join(MODEL_DIR, "autoencoder_threshold.json")
    with open(thresh_path, "w") as f:
        json.dump({
            "threshold":       float(threshold),
            "threshold_sigma": THRESHOLD_SIGMA,
            "features":        FEATURES,
        }, f, indent=2)
    print(f"Threshold saved: {thresh_path}")

    compare_with_isolation_forest(ae_prec, ae_rec, ae_f1)

    print("Done. Next step: wire both models into the FastAPI routes.")


if __name__ == "__main__":
    main()
