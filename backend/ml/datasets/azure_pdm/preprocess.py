"""
Preprocesses the Microsoft Azure Predictive Maintenance dataset.

Inputs  (from data/raw/azure_pdm/):
  PdM_telemetry.csv  — hourly sensor readings per machine (volt, rotate, pressure, vibration)
  PdM_failures.csv   — timestamped component failure events per machine

Output (data/processed/azure_pdm.csv):
  timestamp, machine_id, voltage, rotation, pressure, vibration, failure_within_24h

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 WHY "FAILURE WITHIN 24 HOURS" — NOT "FAILURE AT THIS MOMENT"?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Industrial predictive maintenance has a deceptively simple goal:
 warn operators *before* the machine breaks, not at the exact second
 it breaks.

 If we labeled only the single timestamp where a failure is recorded:
  - The model learns to flag readings only when a failure is already
    happening. By then, the machine has already failed — it's too late
    for a maintenance action to help.
  - We'd get near-perfect "precision" on a useless task.

 Instead we apply a "look-ahead window" approach:
  - For every telemetry row, we look 24 hours *forward* in time for
    that machine. If a failure occurs within that window → label 1.
  - This teaches the model to recognise the *precursor* operating state
    that precedes breakdown, not just the breakdown moment itself.
  - A 24-hour window is a standard industry default: long enough for
    maintenance to be scheduled and carried out, short enough that the
    label still has predictive signal (sensor drift toward failure
    typically accelerates in the 6–24h before breakdown).

 The window length is a tunable business parameter:
  FAILURE_LOOKAHEAD_HOURS = 24   ← adjust based on your maintenance SLA
  Increasing it (e.g. 48h) gives operators more lead time but dilutes
  the label — more rows become "imminent failure" rows, which may hurt
  model precision.  Decreasing it (e.g. 8h) sharpens the boundary but
  gives less advance warning.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CLASS IMBALANCE NOTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 The Azure dataset has 761 failure events across ~876,100 telemetry
 rows. Even with a 24-hour window (24 rows per event → ~18K positive
 rows), the positive class is ~2–3% of the total — similar to AI4I.

 We handle imbalance the same way as the AI4I pipeline:
  - Isolation Forest contamination is set to the *measured* failure rate
    (not the 'auto' default of 10%).
  - The Autoencoder trains ONLY on normal rows so it never learns to
    reconstruct failures — class imbalance is irrelevant for training.
  - We report both precision and recall, noting that recall matters
    more for safety-critical maintenance (a missed failure is worse
    than a false alarm).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
  # from the backend/ directory with venv active:
  python ml/datasets/azure_pdm/preprocess.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(SCRIPT_DIR, "../../../../data/raw/azure_pdm")
PROCESSED_DIR= os.path.join(SCRIPT_DIR, "../../../../data/processed")
OUTPUT_PATH  = os.path.join(PROCESSED_DIR, "azure_pdm.csv")

TELEMETRY_PATH = os.path.join(RAW_DIR, "PdM_telemetry.csv")
FAILURES_PATH  = os.path.join(RAW_DIR, "PdM_failures.csv")

# Look-ahead window for failure labeling (in hours).
# Change this to tune advance-warning lead time vs. label sharpness.
FAILURE_LOOKAHEAD_HOURS = 24


def check_raw_files():
    missing = []
    for path in [TELEMETRY_PATH, FAILURES_PATH]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        print("ERROR: Required raw files not found:")
        for p in missing:
            print(f"  {p}")
        print("\nRun  python ml/datasets/azure_pdm/download_dataset.py  first.")
        sys.exit(1)


def load_telemetry() -> pd.DataFrame:
    """
    Load PdM_telemetry.csv.

    Raw schema:
      datetime     — ISO timestamp, hourly cadence
      machineID    — integer 1..100
      volt         — voltage reading
      rotate       — rotational speed
      pressure     — pressure reading
      vibration    — vibration reading
    """
    print(f"Loading telemetry from {TELEMETRY_PATH} ...")
    df = pd.read_csv(TELEMETRY_PATH, parse_dates=["datetime"])
    df = df.rename(columns={
        "datetime":  "timestamp",
        "machineID": "machine_id",
        "volt":      "voltage",
        "rotate":    "rotation",
        # pressure and vibration keep their names
    })
    df["machine_id"] = df["machine_id"].apply(lambda x: f"machine_{int(x):03d}")
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} telemetry rows across {df['machine_id'].nunique()} machines")
    return df


def load_failures() -> pd.DataFrame:
    """
    Load PdM_failures.csv.

    Raw schema:
      datetime     — timestamp of the failure event
      machineID    — integer 1..100
      failure      — component identifier (comp1, comp2, comp3, comp4)
    """
    print(f"Loading failures from {FAILURES_PATH} ...")
    df = pd.read_csv(FAILURES_PATH, parse_dates=["datetime"])
    df = df.rename(columns={
        "datetime":  "failure_time",
        "machineID": "machine_id",
    })
    df["machine_id"] = df["machine_id"].apply(lambda x: f"machine_{int(x):03d}")
    print(f"  Loaded {len(df):,} failure events across {df['machine_id'].nunique()} machines")
    return df


def create_lookahead_label(telemetry: pd.DataFrame, failures: pd.DataFrame) -> pd.DataFrame:
    """
    For every telemetry row, set failure_within_24h = 1 if there is a failure
    event for the same machine within the next FAILURE_LOOKAHEAD_HOURS hours.

    This is O(N log N) per machine via sorted groupby + searchsorted —
    no quadratic pair-wise comparison.

    Unit-safety note (pandas 3.x):
      pandas 3.x stores datetime columns as datetime64[us] (microseconds).
      pd.Timedelta.value always returns nanoseconds — a 1000x mismatch that
      would make a 24-hour window look like a 24,000-hour window and label
      almost every row as a failure.  We fix this by using numpy timedelta64
      arithmetic directly (unit-aware) and normalising everything to int64
      microseconds before calling searchsorted.
    """
    print(f"\nCreating {FAILURE_LOOKAHEAD_HOURS}h look-ahead failure labels ...")
    # numpy timedelta64 with explicit unit — numpy handles the us/ns conversion
    window_delta = np.timedelta64(FAILURE_LOOKAHEAD_HOURS, "h")

    label_series = []

    for machine_id, tele_grp in telemetry.groupby("machine_id", sort=True):
        fail_grp = failures[failures["machine_id"] == machine_id]

        timestamps = tele_grp["timestamp"].values  # datetime64[us] in pandas 3.x
        fail_times = fail_grp["failure_time"].values

        if len(fail_times) == 0:
            labels = np.zeros(len(tele_grp), dtype=np.int8)
        else:
            fail_times_sorted = np.sort(fail_times)

            # Compute window end via numpy datetime64 arithmetic — unit-aware,
            # no nanosecond/microsecond confusion.
            t_end = timestamps + window_delta

            # Normalise to datetime64[us] → int64 (microseconds since epoch)
            # for both arrays before searchsorted so units are always identical.
            t_int     = timestamps.astype("datetime64[us]").astype("int64")
            t_end_int = t_end.astype("datetime64[us]").astype("int64")
            fail_int  = fail_times_sorted.astype("datetime64[us]").astype("int64")

            # Any failure F in [T, T+window]?
            left_idx  = np.searchsorted(fail_int, t_int,     side="left")
            right_idx = np.searchsorted(fail_int, t_end_int, side="right")
            labels    = (right_idx > left_idx).astype(np.int8)

        label_series.append(
            pd.Series(labels, index=tele_grp.index, name="failure_within_24h")
        )

    all_labels = pd.concat(label_series).sort_index()
    telemetry["failure_within_24h"] = all_labels
    return telemetry


def report_stats(df: pd.DataFrame):
    total      = len(df)
    n_positive = df["failure_within_24h"].sum()
    n_negative = total - n_positive
    rate_pct   = n_positive / total * 100

    print("\n" + "=" * 60)
    print("Dataset Statistics After Preprocessing")
    print("=" * 60)
    print(f"  Total rows            : {total:,}")
    print(f"  Machines              : {df['machine_id'].nunique()}")
    print(f"  Date range            : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Normal rows (label=0) : {n_negative:,}  ({100 - rate_pct:.1f}%)")
    print(f"  Failure rows (label=1): {n_positive:,}  ({rate_pct:.2f}%)")
    print(f"\n  Positive rate  ≈ {rate_pct:.2f}%  (contamination for Isolation Forest)")
    print(
        f"\n  Class imbalance note: {rate_pct:.2f}% positive class is comparable to\n"
        f"  AI4I's 3.4% — same handling strategy applies:\n"
        f"    • Isolation Forest contamination set to {rate_pct/100:.4f}\n"
        f"    • Autoencoder trained on normal-only rows (imbalance irrelevant)\n"
        f"    • Evaluate recall first, precision second (safety context)"
    )
    print("=" * 60)


def main():
    print("=" * 60)
    print("Azure PdM — Preprocessing Pipeline")
    print(f"  Look-ahead window : {FAILURE_LOOKAHEAD_HOURS} hours")
    print("=" * 60 + "\n")

    check_raw_files()
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    telemetry = load_telemetry()
    failures  = load_failures()

    # Create look-ahead labels
    df = create_lookahead_label(telemetry, failures)

    # Select and reorder final columns
    output_cols = [
        "timestamp", "machine_id",
        "voltage", "rotation", "pressure", "vibration",
        "failure_within_24h",
    ]
    df = df[output_cols]

    # Remove any rows with NaN sensor values (shouldn't exist, but defensive)
    before = len(df)
    df = df.dropna(subset=["voltage", "rotation", "pressure", "vibration"])
    if len(df) < before:
        print(f"\n  Dropped {before - len(df)} rows with NaN sensor values.")

    report_stats(df)

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 ** 2)
    print(f"\n✓  Preprocessed data saved to:\n   {OUTPUT_PATH}  ({size_mb:.1f} MB)")
    print("\nNext step:\n  python ml/train_azure_models.py")


if __name__ == "__main__":
    main()
