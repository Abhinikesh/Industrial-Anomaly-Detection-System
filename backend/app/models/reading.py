"""
backend/app/models/reading.py
=============================
Pydantic schemas for ingest payloads and stored readings.

IngestPayload performs machine-type-aware validation:
  - Rejects unknown machine_type values with a clear 422 error.
  - Verifies that all required sensor keys for the given machine_type are
    present in sensor_values, so the model scorer never silently receives
    an incomplete feature vector.
  - Ensures every value in sensor_values is a finite number (no NaN / inf).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator

# ── Known machine types and their required sensor keys ────────────────────────
# Keep in sync with app/config.py:settings.required_sensor_keys and
# app/services/anomaly_service.py:_MACHINE_MODEL_PATHS

KNOWN_MACHINE_TYPES: Dict[str, list[str]] = {
    "milling_machine": [
        "air_temp", "process_temp", "rpm", "torque", "tool_wear",
    ],
    "fleet_machine": [
        "voltage", "rotation", "pressure", "vibration",
    ],
    "water_pump": [
        "sensor_04", "sensor_00", "sensor_10", "sensor_06",
        "sensor_11", "sensor_07", "sensor_02", "sensor_08",
        "sensor_12", "sensor_09", "sensor_01", "sensor_03",
        "sensor_05", "sensor_40", "sensor_48",
    ],
}


class SensorReading(BaseModel):
    """Shape of one fully-scored reading stored in MongoDB.

    sensor_values holds raw readings in a free-form dict so this schema
    works for any machine type without a new Pydantic model per dataset.
    """
    timestamp:    str
    machine_type: str = "milling_machine"
    machine_id:   str = "AI4I-001"

    # raw sensor values — keys are machine-type-specific
    sensor_values: Dict[str, float] = Field(default_factory=dict)

    # ground-truth label from the dataset (offline eval only)
    true_failure:  int = 0

    # model outputs — populated by anomaly_service before saving
    iso_score:  Optional[float] = None
    iso_flag:   Optional[int]   = None   # 1 = anomaly, 0 = normal
    ae_score:   Optional[float] = None
    ae_flag:    Optional[int]   = None
    is_anomaly: Optional[bool]  = None   # True if either model flagged


class IngestPayload(BaseModel):
    """What the simulator (or any edge client) POSTs to /ingest.

    Validates:
    1. machine_type is one of the registered types.
    2. All required sensor keys for that machine_type are present.
    3. All sensor values are finite real numbers (no NaN / inf).
    """
    timestamp:     str
    machine_type:  str              = "milling_machine"
    machine_id:    str              = "AI4I-001"
    sensor_values: Dict[str, float] = Field(default_factory=dict)
    true_failure:  int              = 0

    @model_validator(mode="after")
    def validate_sensor_values(self) -> "IngestPayload":
        mt = self.machine_type

        # 1. Reject unknown machine types immediately
        if mt not in KNOWN_MACHINE_TYPES:
            raise ValueError(
                f"Unknown machine_type '{mt}'. "
                f"Valid types: {list(KNOWN_MACHINE_TYPES)}"
            )

        required = KNOWN_MACHINE_TYPES[mt]
        sv = self.sensor_values

        # 2. Check all required keys are present
        missing = [k for k in required if k not in sv]
        if missing:
            raise ValueError(
                f"sensor_values for machine_type='{mt}' is missing required "
                f"key(s): {missing}. "
                f"All required keys: {required}"
            )

        # 3. Ensure all values are finite (catch NaN / inf from bad simulators)
        bad = [k for k, v in sv.items() if not math.isfinite(v)]
        if bad:
            raise ValueError(
                f"sensor_values contains non-finite (NaN or inf) values for "
                f"key(s): {bad}. All sensor readings must be real numbers."
            )

        return self
