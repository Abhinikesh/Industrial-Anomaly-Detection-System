"""Quick demo: sends 30 rows from the dataset then stops (for verification)."""
import os, sys, time, requests, pandas as pd
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../data/raw/ai4i2020.csv")
INGEST_URL = "http://localhost:8000/ingest"
DELAY      = 0.2
MAX_ROWS   = 30

df = pd.read_csv(DATA_PATH)
print(f"Streaming {MAX_ROWS} rows → {INGEST_URL}\n")

for i in range(MAX_ROWS):
    row     = df.iloc[i]
    payload = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "machine_id":   str(row.get("Product ID", "SIM")),
        "air_temp":     float(row["Air temperature [K]"]),
        "process_temp": float(row["Process temperature [K]"]),
        "rpm":          float(row["Rotational speed [rpm]"]),
        "torque":       float(row["Torque [Nm]"]),
        "tool_wear":    float(row["Tool wear [min]"]),
        "true_failure": int(row["Machine failure"]),
    }
    t0   = time.perf_counter()
    resp = requests.post(INGEST_URL, json=payload, timeout=5)
    ms   = (time.perf_counter() - t0) * 1000
    r    = resp.json()

    anomaly_tag  = " ⚠ ANOMALY"  if r["is_anomaly"]   else ""
    failure_tag  = " [TRUE FAIL]" if payload["true_failure"] else ""
    iso_tag      = " IF✗" if r["iso_flag"] else ""
    ae_tag       = " AE✗" if r["ae_flag"]  else ""

    print(f"[{i:03d}]  T={payload['air_temp']:.1f}K  "
          f"RPM={payload['rpm']:.0f}  "
          f"Torque={payload['torque']:.1f}  "
          f"Wear={payload['tool_wear']:.0f}min"
          f"{iso_tag}{ae_tag}{anomaly_tag}{failure_tag}  "
          f"({ms:.0f}ms)")
    time.sleep(DELAY)

print("\nDemo complete.")
