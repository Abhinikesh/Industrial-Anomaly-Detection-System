"""
Simulates a live multi-machine sensor fleet by replaying the AI4I 2020 dataset
row by row, sending each reading as a POST request to the backend /ingest endpoint.

Changes from original single-machine version
---------------------------------------------
- Every payload now includes machine_type = "milling_machine".
- machine_id is drawn from a fixed pool of 5 IDs (AI4I-001 … AI4I-005),
  cycling with the row index so each machine sees a different slice of readings.
- Raw sensor values are wrapped inside a sensor_values dict to match the
  updated IngestPayload schema (flexible, machine-type-agnostic).

Row order: we keep the ORIGINAL dataset order rather than shuffling.
  - The AI4I dataset was generated with some time-based patterns (tool wear
    increases monotonically, failure modes cluster at certain wear levels).
    Preserving order means the simulator shows realistic wear progression,
    which makes the demo more meaningful than random jumps.
  - The loop cycles back to row 0 when it reaches the end, so the demo
    runs indefinitely without crashing or duplicate key collisions.

Usage:
  python simulator.py              # 1 second between readings (default)
  python simulator.py --fast       # 0.2 seconds (good for quick testing)
  python simulator.py --delay 0.5  # custom delay in seconds
"""

import argparse
import time
import sys
import os
import requests
import pandas as pd
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../data/raw/ai4i2020.csv")

# Default cadence — can be overridden via CLI
SIMULATION_SPEED = 1.0   # seconds

INGEST_URL = "http://localhost:8000/ingest"

# Machine-type tag for every reading sent by this simulator
MACHINE_TYPE = "milling_machine"

# Pool of machine IDs to simulate a small fleet.
# Row i is assigned to MACHINE_POOL[i % len(MACHINE_POOL)] so each machine
# consistently sees the same subset of readings within a cycle.
MACHINE_POOL = [
    "AI4I-001",
    "AI4I-002",
    "AI4I-003",
    "AI4I-004",
    "AI4I-005",
]


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run download_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows — will cycle continuously\n")
    return df


def build_payload(row, row_idx: int) -> dict:
    """Package one dataset row into the dict the /ingest endpoint expects.

    sensor_values is a dict (not flat scalar fields) so the backend can
    handle readings from any registered machine type.
    """
    machine_id = MACHINE_POOL[row_idx % len(MACHINE_POOL)]
    return {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "machine_type": MACHINE_TYPE,
        "machine_id":   machine_id,
        "sensor_values": {
            "air_temp":    float(row["Air temperature [K]"]),
            "process_temp": float(row["Process temperature [K]"]),
            "rpm":          float(row["Rotational speed [rpm]"]),
            "torque":       float(row["Torque [Nm]"]),
            "tool_wear":    float(row["Tool wear [min]"]),
        },
        # Include true label so we can compare against model predictions in the UI
        "true_failure": int(row["Machine failure"]),
    }


def log_line(i: int, payload: dict, resp_ok: bool, elapsed_ms: float, model_info=None):
    ts_short    = datetime.now().strftime("%H:%M:%S")
    sv          = payload["sensor_values"]
    failure_tag = " [TRUE FAILURE]" if payload["true_failure"] else ""
    status_tag  = "✓ OK" if resp_ok else "✗ FAIL"
    anomaly_tag = " ⚠ ANOMALY" if model_info and model_info.get("is_anomaly") else ""

    print(
        f"[{ts_short}]  Row {i:>5} | {payload['machine_id']} | "
        f"T={sv['air_temp']:.1f}K "
        f"RPM={sv['rpm']:.0f} "
        f"Torque={sv['torque']:.1f}Nm "
        f"Wear={sv['tool_wear']:.0f}min"
        f"{anomaly_tag}{failure_tag} "
        f"→ {status_tag} ({elapsed_ms:.0f}ms)"
    )


def stream_readings(df: pd.DataFrame, delay: float):
    """
    Infinite loop: iterate through the dataset, POST each row, sleep, repeat.
    Handles backend connection loss cleanly with automatic retry.
    """
    n = len(df)
    cycle = 0
    row_idx = 0
    consecutive_errors = 0

    print(f"Streaming to {INGEST_URL} (cadence: {delay}s per packet, Ctrl-C to stop)")
    print(f"Machine pool: {MACHINE_POOL}\n")

    while True:
        row = df.iloc[row_idx]
        payload = build_payload(row, row_idx)

        t0 = time.perf_counter()
        resp_ok = False
        model_info = None

        try:
            resp = requests.post(INGEST_URL, json=payload, timeout=4)
            if resp.status_code == 200:
                resp_ok = True
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
                    "⚠ Backend unreachable on :8000. Retrying in background..."
                )
            elif consecutive_errors % 10 == 0:
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"⚠ Still waiting for backend (:8000)... ({consecutive_errors} failed attempts)"
                )

            # Back off slightly while backend is offline to prevent hammering
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
    p = argparse.ArgumentParser(description="Multi-machine sensor stream simulator")
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--fast",
        action="store_true",
        help="0.2 second delay — good for quick end-to-end testing",
    )
    speed.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECS",
        help="Custom delay in seconds (e.g. --delay 0.5)",
    )
    return p.parse_args()


def main():
    args  = parse_args()
    delay = SIMULATION_SPEED  # default 1s

    if args.fast:
        delay = 0.2
        print("Fast mode active — 0.2s delay")
    elif args.delay is not None:
        delay = args.delay
        print(f"Custom delay active — {delay}s")

    df = load_dataset()

    try:
        stream_readings(df, delay)
    except KeyboardInterrupt:
        print("\n[Simulator safely stopped by user.]")


if __name__ == "__main__":
    main()
