"""
Trains an Isolation Forest on the AI4I 2020 sensor data.

Feature choice: we use the 5 raw numeric sensor columns only.
  - "Type" (L/M/H product quality) is dropped — it's a manufacturing
    process label, not a sensor reading. Isolation Forest works on
    continuous sensor space; encoding a categorical here adds noise
    without clear benefit for unsupervised anomaly detection.
  - UDI / Product ID are just identifiers — dropped.

Run order: download_dataset.py → explore_data.py → this script
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed when saving files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, "../../data/raw/ai4i2020.csv")
MODEL_DIR    = os.path.join(SCRIPT_DIR, "../../models")
RESULTS_DIR  = os.path.join(SCRIPT_DIR, "../../results/isolation_forest")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# contamination: we set this to 0.034 (≈ 3.4%, the actual failure rate found
# in EDA) rather than 'auto'. 'auto' internally uses 0.1 which would
# over-flag too many normal readings as anomalies.  Matching the real
# base rate gives the decision boundary a better prior.
CONTAMINATION = 0.034


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run download_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows from {DATA_PATH}")
    return df


def prepare_features(df):
    X = df[FEATURES].copy()
    y = df["Machine failure"].values

    # Scaling matters here: rotational speed can be 1000-3000 rpm while
    # torque sits in the 3-80 Nm range.  Without scaling, the high-magnitude
    # RPM column dominates the random splits inside each isolation tree,
    # effectively making the other sensors invisible.  StandardScaler puts
    # every feature on mean=0, std=1 so each sensor contributes equally.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def train(X_scaled):
    print(f"\nTraining Isolation Forest (contamination={CONTAMINATION}, n_estimators=100)...")
    clf = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,   # use all cores, speeds up on larger datasets
    )
    clf.fit(X_scaled)
    return clf


def evaluate(clf, X_scaled, y_true):
    """
    Returns predictions and raw anomaly scores.
    sklearn convention: predict() returns -1 (anomaly) or +1 (normal).
    We flip to 0/1 so it aligns with the dataset's "Machine failure" label.
    """
    raw_preds  = clf.predict(X_scaled)           # -1 or +1
    is_anomaly = (raw_preds == -1).astype(int)   # 0 = normal, 1 = anomaly

    # score_samples returns negative average path length (higher = more normal)
    # we negate it so "higher score = more anomalous", easier to reason about
    scores = -clf.score_samples(X_scaled)

    return is_anomaly, scores


def print_results(y_true, y_pred):
    print("\n" + "=" * 58)
    print("Isolation Forest — Evaluation Results")
    print("=" * 58)

    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"\n  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1-score  : {f1:.3f}")

    print("\nFull classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Failure"]))

    print("What these numbers mean for this use case:")
    print("""
  - RECALL matters most here. A missed real failure (false negative)
    means a machine runs until it breaks — expensive and dangerous.
    A false alarm (false positive) just triggers an unnecessary check,
    which is annoying but cheap.

  - PRECISION matters too, but less urgently. Too many false alarms
    erodes operator trust and leads them to ignore alerts entirely
    (the "cry wolf" problem).

  - For production use, a reasonable target is recall > 0.70 while
    keeping precision above 0.30. Isolation Forest is unsupervised
    (it never sees the failure labels during training), so these
    numbers are genuinely informative about the score threshold.
""")
    return prec, rec, f1


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Normal", "Failure"]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual",    fontsize=11)
    ax.set_title("Isolation Forest — Confusion Matrix", fontsize=12, pad=10)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_score_distribution(scores, y_true):
    """
    Histogram of anomaly scores split by actual label.
    A good model should show clearly separated distributions.
    """
    normal_scores  = scores[y_true == 0]
    failure_scores = scores[y_true == 1]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.hist(normal_scores,  bins=60, alpha=0.65, color="#4a90d9",
            label=f"Normal  (n={len(normal_scores):,})",  density=True)
    ax.hist(failure_scores, bins=60, alpha=0.65, color="#e05252",
            label=f"Failure (n={len(failure_scores):,})", density=True)

    # draw the decision threshold line (score at contamination quantile)
    threshold = np.quantile(scores, 1 - CONTAMINATION)
    ax.axvline(threshold, color="orange", linestyle="--", linewidth=1.5,
               label=f"Decision threshold ({threshold:.3f})")

    ax.set_xlabel("Anomaly Score (higher = more anomalous)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Isolation Forest — Score Distribution", fontsize=12, pad=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "score_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def save_models(clf, scaler):
    iso_path    = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    joblib.dump(clf,    iso_path)
    joblib.dump(scaler, scaler_path)

    print(f"\nModels saved:")
    print(f"  {iso_path}")
    print(f"  {scaler_path}")


def main():
    df            = load_data()
    X_scaled, y, scaler = prepare_features(df)
    clf           = train(X_scaled)
    y_pred, scores = evaluate(clf, X_scaled, y)

    print_results(y, y_pred)

    print("\nGenerating plots...")
    save_confusion_matrix(y, y_pred)
    save_score_distribution(scores, y)

    save_models(clf, scaler)

    print("\nDone. Next step: train_autoencoder.py")


if __name__ == "__main__":
    main()
