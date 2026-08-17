"""
Simulates a live sensor stream by replaying the AI4I 2020 dataset row by row,
sending each reading as a POST request to the backend /ingest endpoint.

Row order: we keep the ORIGINAL dataset order rather than shuffling.
  - The AI4I dataset was generated with some time-based patterns (tool wear
    increases monotonically, failure modes cluster at certain wear levels).
    Preserving order means the simulator shows realistic wear progression,
    which makes the demo more meaningful than random jumps.
  - The loop cycles back to row 0 when it reaches the end, so the demo
    runs indefinitely.

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

# how long to wait between readings (can be overridden via CLI)
SIMULATION_SPEED = 1.0   # seconds

INGEST_URL = "http://localhost:8000/ingest"

SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Dataset not found at {DATA_PATH}")
        print("Run download_dataset.py first.")
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows — will cycle continuously\n")
    return df


def build_payload(row):
    """Package one dataset row into the dict the /ingest endpoint expects."""
    return {
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "machine_id":  row.get("Product ID", "SIM-001"),
        "air_temp":    float(row["Air temperature [K]"]),
        "process_temp":float(row["Process temperature [K]"]),
        "rpm":         float(row["Rotational speed [rpm]"]),
        "torque":      float(row["Torque [Nm]"]),
        "tool_wear":   float(row["Tool wear [min]"]),
        # include true label so we can compare against model predictions in the UI
        "true_failure": int(row["Machine failure"]),
    }


def log_line(i, payload, resp_ok, elapsed_ms):
    ts_short   = datetime.now().strftime("%H:%M:%S")
    failure_tag = " ⚠ FAILURE" if payload["true_failure"] else ""
    status_tag  = "✓" if resp_ok else "✗"
    print(
        f"[{ts_short}]  row {i:>5}  "
        f"T={payload['air_temp']:.1f}K  "
        f"RPM={payload['rpm']:.0f}  "
        f"Torque={payload['torque']:.1f}Nm  "
        f"Wear={payload['tool_wear']:.0f}min"
        f"{failure_tag}  "
        f"→ {status_tag} {elapsed_ms:.0f}ms"
    )


def stream_readings(df, delay):
    """
    Infinite loop: iterate through the dataset, POST each row, sleep, repeat.
    Ctrl-C exits cleanly.
    """
    n        = len(df)
    cycle    = 0
    row_idx  = 0

    print(f"Streaming to {INGEST_URL}  (delay={delay}s, Ctrl-C to stop)\n")

    while True:
        row     = df.iloc[row_idx]
        payload = build_payload(row)

        t0 = time.perf_counter()
        try:
            resp    = requests.post(INGEST_URL, json=payload, timeout=5)
            resp_ok = resp.status_code == 200
        except requests.exceptions.ConnectionError:
            resp_ok = False
            # don't spam the console if the backend isn't up yet
            if row_idx == 0:
                print("  Backend not reachable — start FastAPI first (uvicorn app.main:app --reload)")
        except requests.exceptions.Timeout:
            resp_ok = False

        elapsed_ms = (time.perf_counter() - t0) * 1000
        log_line(row_idx, payload, resp_ok, elapsed_ms)

        row_idx += 1
        if row_idx >= n:
            row_idx = 0
            cycle  += 1
            print(f"\n--- cycle {cycle} complete, looping back to row 0 ---\n")

        time.sleep(delay)


def parse_args():
    p = argparse.ArgumentParser(description="Sensor stream simulator")
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--fast",
        action="store_true",
        help="0.2 second delay — good for quick end-to-end testing"
    )
    speed.add_argument(
        "--delay",
        type=float,
        default=None,
        metavar="SECS",
        help="Custom delay in seconds (e.g. --delay 0.5)"
    )
    return p.parse_args()


def main():
    args  = parse_args()
    delay = SIMULATION_SPEED  # default

    if args.fast:
        delay = 0.2
        print("Fast mode — 0.2s delay")
    elif args.delay is not None:
        delay = args.delay
        print(f"Custom delay — {delay}s")

    df = load_dataset()

    try:
        stream_readings(df, delay)
    except KeyboardInterrupt:
        print("\nSimulator stopped.")


if __name__ == "__main__":
    main()
