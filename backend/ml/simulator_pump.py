"""
Simulates the industrial water pump by replaying the preprocessed pump sensor
data chronologically and streaming each reading to the /ingest endpoint.

machine_type : "water_pump"
machine_id   : "pump_001"  (single pump unit in this dataset)

WHY CHRONOLOGICAL ORDER MATTERS FOR THIS DATASET
─────────────────────────────────────────────────
Unlike the AI4I and Azure datasets (where readings from different machines are
largely independent), the pump sensor dataset captures a single pump's degradation
trajectory over several months.  The machine_status transitions are:

    NORMAL → ... → NORMAL → RECOVERING → NORMAL → RECOVERING → BROKEN → RECOVERING → NORMAL

If we shuffle or randomise the row order, we lose the most valuable property of
this dataset: the gradual sensor drift that precedes each failure event.
By replaying strictly in timestamp order, the live anomaly score in the dashboard
will visibly ramp up as the pump approaches each RECOVERING/BROKEN episode —
this is the core value proposition of predictive maintenance.

Usage:
  python ml/simulator_pump.py              # 1s cadence
  python ml/simulator_pump.py --fast       # 0.1s cadence — good for quick tests
  python ml/simulator_pump.py --delay 0.5  # custom delay
  python ml/simulator_pump.py --repeat     # loop back to start after reaching end
"""

import argparse
import time
import sys
import os
import requests
import pandas as pd
from datetime import datetime, timezone

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROCESSED_PATH = os.path.join(SCRIPT_DIR, "../../data/processed/pump_sensor.csv")

INGEST_URL   = "http://localhost:8000/ingest"
MACHINE_TYPE = "water_pump"
MACHINE_ID   = "pump_001"
DEFAULT_DELAY= 1.0   # seconds between readings


def load_dataset() -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(PROCESSED_PATH):
        print(f"ERROR: Preprocessed file not found at:\n  {PROCESSED_PATH}")
        print(
            "\nRun:\n"
            "  python ml/datasets/pump_sensor/preprocess.py"
        )
        sys.exit(1)

    print(f"Loading pump sensor data from:\n  {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["timestamp"])

    # Preserve strict chronological order — this is the entire point for this dataset
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Sensor columns are everything that starts with "sensor_"
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    status_counts = df["machine_status"].value_counts().to_dict()
    print(
        f"  Rows            : {len(df):,}\n"
        f"  Sensors         : {len(sensor_cols)}\n"
        f"  Date range      : {df['timestamp'].min()} → {df['timestamp'].max()}\n"
        f"  machine_status  : {status_counts}\n"
        f"  Replaying in strict chronological order.\n"
    )
    return df, sensor_cols


def build_payload(row: pd.Series, sensor_cols: list[str]) -> dict:
    """Package one pump sensor row into the /ingest payload format."""
    return {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "machine_type": MACHINE_TYPE,
        "machine_id":   MACHINE_ID,
        "sensor_values": {col: float(row[col]) for col in sensor_cols},
        # Include ground-truth status for offline evaluation
        "true_status":  str(row.get("machine_status", "UNKNOWN")),
        "true_label":   int(row.get("label", 0)),
    }


def status_icon(machine_status: str) -> str:
    return {"NORMAL": "✓", "RECOVERING": "↑", "BROKEN": "✗"}.get(machine_status, "?")


def log_line(i: int, payload: dict, resp_ok: bool, elapsed_ms: float, model_info=None):
    ts_short    = datetime.now().strftime("%H:%M:%S")
    status      = payload.get("true_status", "UNKNOWN")
    icon        = status_icon(status)
    anomaly_tag = " ⚠ ANOMALY" if model_info and model_info.get("is_anomaly") else ""
    api_tag     = "✓" if resp_ok else "✗ FAIL"

    # Show a representative subset of sensor values to keep output readable
    sv = payload["sensor_values"]
    keys = list(sv.keys())[:4]   # first 4 sensors
    sensor_str = "  ".join(f"{k}={sv[k]:.2f}" for k in keys) + " …"

    print(
        f"[{ts_short}]  Row {i:>6} | {icon} {status:<11} | "
        f"{sensor_str}{anomaly_tag} → {api_tag} ({elapsed_ms:.0f}ms)"
    )


def stream_readings(df: pd.DataFrame, sensor_cols: list[str],
                    delay: float, repeat: bool):
    n = len(df)
    consecutive_errors = 0
    cycle = 0

    print(f"Streaming water pump to {INGEST_URL}")
    print(f"(cadence: {delay}s per packet, Ctrl-C to stop)\n")

    while True:
        for row_idx in range(n):
            row     = df.iloc[row_idx]
            payload = build_payload(row, sensor_cols)

            t0         = time.perf_counter()
            resp_ok    = False
            model_info = None

            try:
                resp = requests.post(INGEST_URL, json=payload, timeout=4)
                if resp.status_code == 200:
                    resp_ok    = True
                    model_info = resp.json()
                    if consecutive_errors > 0:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                              "✓ Connection restored. Resuming stream.")
                        consecutive_errors = 0
                else:
                    consecutive_errors += 1
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                consecutive_errors += 1
                if consecutive_errors == 1:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          "⚠ Backend unreachable on :8000. Retrying...")
                elif consecutive_errors % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"⚠ Still waiting... ({consecutive_errors} attempts)")
                time.sleep(2.0)
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            log_line(row_idx, payload, resp_ok, elapsed_ms, model_info)
            time.sleep(delay)

        # End of dataset
        cycle += 1
        if not repeat:
            print(f"\n[Pump simulator] Reached end of dataset ({n:,} rows). "
                  f"Use --repeat to loop continuously.")
            break
        print(f"\n--- [Cycle {cycle} complete: looping back to row 0] ---\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Industrial water pump sensor stream simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ml/simulator_pump.py\n"
            "  python ml/simulator_pump.py --fast\n"
            "  python ml/simulator_pump.py --fast --repeat\n"
        ),
    )
    speed = p.add_mutually_exclusive_group()
    speed.add_argument("--fast",  action="store_true",
                       help="0.1 second delay — fast replay for quick testing")
    speed.add_argument("--delay", type=float, default=None, metavar="SECS",
                       help="Custom delay in seconds (e.g. --delay 0.5)")
    p.add_argument("--repeat", action="store_true",
                   help="Loop back to row 0 after reaching the end of the dataset")
    return p.parse_args()


def main():
    args  = parse_args()
    delay = DEFAULT_DELAY

    if args.fast:
        delay = 0.1
        print("Fast mode — 0.1s delay")
    elif args.delay is not None:
        delay = args.delay
        print(f"Custom delay — {delay}s")

    df, sensor_cols = load_dataset()

    try:
        stream_readings(df, sensor_cols, delay, repeat=args.repeat)
    except KeyboardInterrupt:
        print("\n[Pump simulator safely stopped by user.]")


if __name__ == "__main__":
    main()
