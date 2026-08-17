from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class SensorReading(BaseModel):
    """Shape of one sensor reading stored in MongoDB."""
    timestamp:          str
    machine_id:         str = "SIM-001"

    # raw sensor values
    air_temp:           float
    process_temp:       float
    rpm:                float
    torque:             float
    tool_wear:          float

    # ground-truth label from the dataset (kept for offline evaluation only;
    # a real deployment wouldn't have this available ahead of time)
    true_failure:       int = 0

    # model outputs — populated by anomaly_service before saving
    iso_score:          Optional[float] = None
    iso_flag:           Optional[int]   = None   # 1 = anomaly, 0 = normal

    ae_score:           Optional[float] = None
    ae_flag:            Optional[int]   = None

    is_anomaly:         Optional[bool]  = None   # True if either model flagged


class IngestPayload(BaseModel):
    """What the simulator POSTs to /ingest."""
    timestamp:    str
    machine_id:   str   = "SIM-001"
    air_temp:     float
    process_temp: float
    rpm:          float
    torque:       float
    tool_wear:    float
    true_failure: int   = 0
