"""
Exploratory data analysis on the AI4I 2020 dataset.
Run after download_dataset.py has saved data/raw/ai4i2020.csv.

Outputs:
  - Console summary (rows, dtypes, missing values, failure stats, per-sensor stats)
  - results/eda/correlation_heatmap.png
  - results/eda/sensor_distributions.png
  - results/eda/failure_type_breakdown.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../data/raw/ai4i2020.csv")
EDA_DIR    = os.path.join(SCRIPT_DIR, "../../results/eda")

os.makedirs(EDA_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

# ── column reference ───────────────────────────────────────────────────────────
COL_DOCS = {
    "UDI":                 "Unique identifier (1–10000)",
    "Product ID":          "Product quality variant code (L/M/H prefix)",
    "Type":                "Product type: L=Low, M=Medium, H=High quality",
    "Air temperature [K]": "Ambient air temperature in Kelvin",
    "Process temperature [K]": "Process temperature in Kelvin (usually ~10 K above air temp)",
    "Rotational speed [rpm]": "Spindle rotational speed in RPM",
    "Torque [Nm]":         "Torque applied to the tool in Newton-metres",
    "Tool wear [min]":     "Cumulative tool usage time in minutes",
    "Machine failure":     "Binary label: 1 = some failure occurred, 0 = normal",
    "TWF":                 "Tool Wear Failure — tool reached wear limit",
    "HDF":                 "Heat Dissipation Failure — temp diff too small at high RPM",
    "PWF":                 "Power Failure — torque × RPM outside valid power range",
    "OSF":                 "Overstrain Failure — tool wear × torque exceeds limit",
    "RNF":                 "Random Failure — 0.1% random chance, no physical cause",
}

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Run download_dataset.py first."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def print_overview(df):
    print("=" * 60)
    print("AI4I 2020 Predictive Maintenance — Dataset Overview")
    print("=" * 60)
    print(f"\nRows: {len(df):,}   Columns: {len(df.columns)}")

    print("\n── Column reference ──────────────────────────────────────")
    for col, desc in COL_DOCS.items():
        marker = "✓" if col in df.columns else "?"
        print(f"  {marker}  {col:<35} {desc}")

    print("\n── Data types ────────────────────────────────────────────")
    print(df.dtypes.to_string())

    missing = df.isnull().sum()
    if missing.any():
        print("\n── Missing values ────────────────────────────────────────")
        print(missing[missing > 0].to_string())
    else:
        print("\nNo missing values.")


def print_sensor_stats(df):
    print("\n── Sensor statistics (all rows) ──────────────────────────")
    stats = df[SENSOR_COLS].describe().T[["mean", "std", "min", "max"]]
    stats = stats.round(2)
    print(stats.to_string())


def print_failure_stats(df):
    total    = len(df)
    failures = df["Machine failure"].sum()
    normal   = total - failures
    pct      = failures / total * 100

    print(f"\n── Failure breakdown ─────────────────────────────────────")
    print(f"  Normal rows:  {normal:,}  ({100-pct:.1f}%)")
    print(f"  Failure rows: {failures:,}  ({pct:.1f}%)")
    print(f"\n  NOTE: ~{pct:.1f}% failure rate is expected and intentional.")
    print("  Anomaly detection datasets are naturally imbalanced.")
    print("  The models need to learn 'normal' from the majority class.\n")

    print("  Failure type counts:")
    for ft in FAILURE_TYPES:
        n = df[ft].sum()
        print(f"    {ft}: {n:4d} cases  ({n/total*100:.2f}%)")


def print_sensor_diff(df):
    """Shows mean sensor values split by normal vs failure — tells us which sensors diverge most."""
    normal_df  = df[df["Machine failure"] == 0][SENSOR_COLS]
    failure_df = df[df["Machine failure"] == 1][SENSOR_COLS]

    diff = pd.DataFrame({
        "Normal mean":  normal_df.mean(),
        "Failure mean": failure_df.mean(),
    })
    diff["Δ (abs)"] = (diff["Failure mean"] - diff["Normal mean"]).abs()
    diff["Δ (%)"]   = (diff["Δ (abs)"] / diff["Normal mean"] * 100).round(1)
    diff = diff.round(3).sort_values("Δ (%)", ascending=False)

    print("\n── Sensor means: normal vs failure ───────────────────────")
    print(diff.to_string())
    return diff


# ── plots ──────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df):
    corr = df[SENSOR_COLS + ["Machine failure"]].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # hide upper triangle to reduce clutter
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, square=True,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Sensor Correlation Heatmap (incl. Machine Failure)", pad=12)
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


def plot_sensor_distributions(df):
    """Overlaid histograms — normal (blue) vs failure (red) for each sensor."""
    normal_df  = df[df["Machine failure"] == 0]
    failure_df = df[df["Machine failure"] == 1]

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    plot_sensors = [
        ("Air temperature [K]",       gs[0, 0]),
        ("Process temperature [K]",   gs[0, 1]),
        ("Rotational speed [rpm]",    gs[0, 2]),
        ("Torque [Nm]",               gs[1, 0]),
        ("Tool wear [min]",           gs[1, 1]),
    ]

    for col, pos in plot_sensors:
        ax = fig.add_subplot(pos)
        ax.hist(normal_df[col],  bins=40, alpha=0.6, color="#4a90d9", label="Normal",  density=True)
        ax.hist(failure_df[col], bins=40, alpha=0.6, color="#e05252", label="Failure", density=True)
        short_name = col.split(" [")[0]
        ax.set_title(short_name, fontsize=10)
        ax.set_xlabel(col.split("[")[1].rstrip("]") if "[" in col else "", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=8)

    fig.suptitle("Sensor Distributions: Normal vs Failure Cases", fontsize=13, y=1.01)
    path = os.path.join(EDA_DIR, "sensor_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_failure_type_breakdown(df):
    counts = {ft: df[ft].sum() for ft in FAILURE_TYPES}
    labels = list(counts.keys())
    vals   = list(counts.values())
    colors = ["#e05252", "#e08a52", "#d4c54a", "#52b0e0", "#a552e0"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_title("Failure Type Distribution", fontsize=12)
    ax.set_ylabel("Number of cases")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(v), ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(EDA_DIR, "failure_type_breakdown.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_summary(diff):
    print("\n" + "=" * 60)
    print("Summary: What the EDA revealed")
    print("=" * 60)
    top = diff.index[0]
    second = diff.index[1]
    print(f"""
  - The dataset has 10,000 readings with ~3.4% labeled as failures.
    This class imbalance is intentional — exactly like real factory data.

  - '{top}' and '{second}' show the largest relative difference
    between normal and failure readings, making them the most
    informative features for the anomaly models.

  - Torque and rotational speed are negatively correlated (faster
    spinning → less torque), which is physically expected. This
    correlation shifts during failures — useful for the Autoencoder
    to learn as a "normal pattern."

  - Tool wear alone isn't a strong failure predictor, but in
    combination with high torque (OSF failure mode) it is.
    Isolation Forest should pick this interaction up.

  - HDF (heat dissipation) is the most common specific failure type.
    It's driven by the temp difference between air and process sensors
    being too small at high RPM — worth checking in the heatmap.

  Next step: feed these 5 sensor columns into Isolation Forest +
  Autoencoder for anomaly scoring. No label info used during training.
""")


def main():
    df   = load_data()
    print_overview(df)
    print_sensor_stats(df)
    print_failure_stats(df)
    diff = print_sensor_diff(df)

    print("\nGenerating plots...")
    plot_correlation_heatmap(df)
    plot_sensor_distributions(df)
    plot_failure_type_breakdown(df)

    print_summary(diff)


if __name__ == "__main__":
    main()
