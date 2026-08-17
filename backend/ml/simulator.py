"""
Simulates a machine sensor stream - generates readings every second
with occasional injected anomalies so we have something to detect.
"""

import numpy as np
import time
import random
from datetime import datetime

# Normal operating ranges for each sensor
NORMAL_RANGES = {
    "temperature": (60, 80),    # Celsius
    "vibration":   (0.1, 0.5),  # g
    "pressure":    (100, 120),  # bar
    "torque":      (40, 60),    # Nm
    "tool_wear":   (0, 200),    # minutes of use
}

def normal_reading():
    """Generate one plausible normal sensor reading."""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "temperature": round(random.uniform(*NORMAL_RANGES["temperature"]), 2),
        "vibration":   round(random.uniform(*NORMAL_RANGES["vibration"]), 3),
        "pressure":    round(random.uniform(*NORMAL_RANGES["pressure"]), 2),
        "torque":      round(random.uniform(*NORMAL_RANGES["torque"]), 2),
        "tool_wear":   round(random.uniform(*NORMAL_RANGES["tool_wear"]), 1),
        "machine_id":  "machine_001",
    }

def anomalous_reading():
    """Spike one or two sensors way outside normal range."""
    reading = normal_reading()
    # randomly pick which sensors to break
    broken = random.sample(list(NORMAL_RANGES.keys()), k=random.randint(1, 2))
    if "temperature" in broken:
        reading["temperature"] = round(random.uniform(110, 140), 2)
    if "vibration" in broken:
        reading["vibration"] = round(random.uniform(1.5, 3.0), 3)
    if "pressure" in broken:
        reading["pressure"] = round(random.uniform(150, 200), 2)
    if "torque" in broken:
        reading["torque"] = round(random.uniform(100, 150), 2)
    return reading

def stream(anomaly_prob=0.05, delay=1.0):
    """
    Infinite generator - yields one reading per `delay` seconds.
    anomaly_prob controls how often a bad reading appears.
    """
    while True:
        if random.random() < anomaly_prob:
            yield anomalous_reading()
        else:
            yield normal_reading()
        time.sleep(delay)


# quick sanity check
if __name__ == "__main__":
    print("Streaming 5 readings (1s apart)...")
    gen = stream(anomaly_prob=0.3, delay=1.0)
    for _ in range(5):
        print(next(gen))
