"""
Simulates a live Azure PdM fleet by replaying preprocessed Azure telemetry,
sending each reading as a POST request to the backend /ingest endpoint.

This runs alongside simulator.py — both can POST to the same /ingest endpoint
simultaneously; the backend dispatches each reading to its correct model pair
based on machine_type.

machine_type sent by this simulator : "fleet_machine"
machine_ids                         : machine_001 … machine_010  (configurable)

Dataset source:
  Preprocessed output of  ml/datasets/azure_pdm/preprocess.py
  at  data/processed/azure_pdm.csv

Differences from simulator.py (AI4I):
  - Reads 4 sensors instead of 5: voltage, rotation, pressure, vibration
  - machine_type is "fleet_machine" (maps to Azure models in anomaly_service)
  - machine_id comes directly from the dataset (machine_001…machine_100)
  - Default replay is filtered to a configurable subset of machines (1-10)
    so the demo doesn't overwhelm a single terminal with 100 parallel streams
  - Row ordering preserves the original chronological order per machine

Usage:
  python ml/simulator_azure.py              # machines 1-10, 1s cadence
  python ml/simulator_azure.py --fast       # 0.2s cadence (quick test)
  python ml/simulator_azure.py --delay 0.5  # custom delay
  python ml/simulator_azure.py --machines 1,5,10,42  # specific machine IDs
"""

import argparse
import time
import sys
import os
import requests
import pandas as pd
from datetime import datetime, timezone

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH  = os.path.join(SCRIPT_DIR, "../../data/processed/azure_pdm.csv")

SIMULATION_SPEED = 1.0   # seconds between readings
INGEST_URL       = "http://localhost:8000/ingest"
MACHINE_TYPE     = "fleet_machine"

# Default machines to simulate (IDs from the dataset: machine_001 … machine_100)
# We use 1-10 for the demo fleet. Adjust or pass --machines to change.
DEFAULT_MACHINE_IDS = [f"machine_{i:03d}" for i in range(1, 11)]


def load_dataset(machine_ids: list) -> pd.DataFrame:
    if not os.path.exists(PROCESSED_PATH):
        print(f"ERROR: Preprocessed file not found at:\n  {PROCESSED_PATH}")
        print(
            "\nRun these two steps first:\n"
            "  python ml/datasets/azure_pdm/download_dataset.py\n"
            "  python ml/datasets/azure_pdm/preprocess.py"
        )
        sys.exit(1)

    print(f"Loading Azure PdM telemetry from:\n  {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])

    # Filter to the requested machine subset
    df = df[df["machine_id"].isin(machine_ids)].copy()

    if df.empty:
        print(f"ERROR: No rows found for machine_ids {machine_ids}")
        print(f"Available IDs: {pd.read_csv(PROCESSED_PATH)['machine_id'].unique()[:10].tolist()} ...")
        sys.exit(1)

    # Interleave all machines by timestamp so readings arrive in chronological
    # order (as they would from real parallel edge sensors)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(
        f"Loaded {len(df):,} rows for {df['machine_id'].nunique()} machines "
        f"({', '.join(sorted(df['machine_id'].unique())[:5])}{'…' if len(machine_ids) > 5 else ''})"
        f"\nWill cycle continuously.\n"
    )
    return df


def build_payload(row: pd.Series) -> dict:
    """Package one Azure PdM telemetry row into the /ingest payload format."""
    return {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "machine_type": MACHINE_TYPE,
        "machine_id":   str(row["machine_id"]),
        "sensor_values": {
            "voltage":   float(row["voltage"]),
            "rotation":  float(row["rotation"]),
            "pressure":  float(row["pressure"]),
            "vibration": float(row["vibration"]),
        },
        # failure_within_24h acts as the ground-truth label for offline evaluation
        "true_failure": int(row["failure_within_24h"]),
    }


def log_line(i: int, payload: dict, resp_ok: bool, elapsed_ms: float, model_info=None):
    ts_short    = datetime.now().strftime("%H:%M:%S")
    sv          = payload["sensor_values"]
    failure_tag = " [IMMINENT FAILURE]" if payload["true_failure"] else ""
    status_tag  = "✓ OK" if resp_ok else "✗ FAIL"
    anomaly_tag = " ⚠ ANOMALY" if model_info and model_info.get("is_anomaly") else ""

    print(
        f"[{ts_short}]  Row {i:>6} | {payload['machine_id']} | "
        f"V={sv['voltage']:.1f}  "
        f"RPM={sv['rotation']:.1f}  "
        f"P={sv['pressure']:.1f}  "
        f"Vib={sv['vibration']:.2f}"
        f"{anomaly_tag}{failure_tag} "
        f"→ {status_tag} ({elapsed_ms:.0f}ms)"
    )


def stream_readings(df: pd.DataFrame, delay: float):
    """
    Infinite loop: iterate through the sorted telemetry, POST each row, sleep, repeat.
    Auto-retries on backend connection loss with back-off.
    """
    n    = len(df)
    cycle = 0
    row_idx = 0
    consecutive_errors = 0

    print(f"Streaming Azure PdM fleet to {INGEST_URL}")
    print(f"(cadence: {delay}s per packet, Ctrl-C to stop)\n")

    while True:
        row     = df.iloc[row_idx]
        payload = build_payload(row)

        t0        = time.perf_counter()
        resp_ok   = False
        model_info = None

        try:
            resp = requests.post(INGEST_URL, json=payload, timeout=4)
            if resp.status_code == 200:
                resp_ok    = True
                model_info = resp.json()
                if consecutive_errors > 0:
                    print(
                        f"[{datetime.now().strftime('%H:%M:%S')}] "
                        "✓ Connection to backend restored. Resuming stream."
                    )
                    consecutive_errors = 0
            else:
                consecutive_errors += 1
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚠ Backend returned HTTP {resp.status_code}"
                )
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            consecutive_errors += 1
            if consecutive_errors == 1:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    "⚠ Backend unreachable on :8000. Retrying..."
                )
            elif consecutive_errors % 10 == 0:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚠ Still waiting for backend... ({consecutive_errors} attempts)"
                )
            time.sleep(2.0)
            continue

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_line(row_idx, payload, resp_ok, elapsed_ms, model_info)

        row_idx += 1
        if row_idx >= n:
            row_idx = 0
            cycle += 1
            print(f"\n--- [Cycle {cycle} Completed: looped back to row 0] ---\n")

        time.sleep(delay)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Azure PdM fleet sensor stream simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ml/simulator_azure.py\n"
            "  python ml/simulator_azure.py --fast\n"
            "  python ml/simulator_azure.py --machines 1,5,10,42\n"
        ),
    )
    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--fast",  action="store_true",
                       help="0.2 second delay — good for quick end-to-end testing")
    speed.add_argument("--delay", type=float, default=None, metavar="SECS",
                       help="Custom delay in seconds (e.g. --delay 0.5)")
    p.add_argument(
        "--machines",
        type=str,
        default=None,
        metavar="IDS",
        help=(
            "Comma-separated list of integer machine IDs to simulate "
            "(e.g. --machines 1,5,10,42).  Defaults to machines 1-10."
        ),
    )
    return p.parse_args()


def resolve_machine_ids(machines_arg: str | None) -> list:
    if machines_arg is None:
        return DEFAULT_MACHINE_IDS
    try:
        ids = [f"machine_{int(x.strip()):03d}" for x in machines_arg.split(",")]
    except ValueError:
        print(f"ERROR: --machines must be comma-separated integers, got: {machines_arg}")
        sys.exit(1)
    return ids


def main():
    args  = parse_args()
    delay = SIMULATION_SPEED

    if args.fast:
        delay = 0.2
        print("Fast mode active — 0.2s delay")
    elif args.delay is not None:
        delay = args.delay
        print(f"Custom delay active — {delay}s")

    machine_ids = resolve_machine_ids(args.machines)
    print(f"Simulating fleet: {machine_ids}\n")

    df = load_dataset(machine_ids)

    try:
        stream_readings(df, delay)
    except KeyboardInterrupt:
        print("\n[Azure simulator safely stopped by user.]")


if __name__ == "__main__":
    main()
