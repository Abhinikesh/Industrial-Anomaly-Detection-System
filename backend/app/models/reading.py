from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime


class SensorReading(BaseModel):
    """Shape of one fully-scored reading stored in MongoDB.

    sensor_values holds the raw sensor readings in a free-form dict so this
    schema works for any machine type (AI4I milling machines, compressors,
    pumps, etc.) without requiring a new Pydantic model per dataset.
    """
    timestamp:     str
    machine_type:  str = "milling_machine"
    machine_id:    str = "AI4I-001"

    # raw sensor values — keys are machine-type-specific
    # e.g. milling_machine: {"air_temp": 298.1, "process_temp": 308.6,
    #                         "rpm": 1551.0, "torque": 42.8, "tool_wear": 0.0}
    sensor_values: Dict[str, float] = Field(default_factory=dict)

    # ground-truth label from the dataset (kept for offline evaluation only;
    # a real deployment wouldn't have this available ahead of time)
    true_failure:  int = 0

    # model outputs — populated by anomaly_service before saving
    iso_score:     Optional[float] = None
    iso_flag:      Optional[int]   = None   # 1 = anomaly, 0 = normal

    ae_score:      Optional[float] = None
    ae_flag:       Optional[int]   = None

    is_anomaly:    Optional[bool]  = None   # True if either model flagged


class IngestPayload(BaseModel):
    """What the simulator (or any edge client) POSTs to /ingest.

    sensor_values is a free-form dict so new machine types can send any
    set of sensor keys without requiring backend schema changes.
    """
    timestamp:     str
    machine_type:  str             = "milling_machine"
    machine_id:    str             = "AI4I-001"
    sensor_values: Dict[str, float] = Field(default_factory=dict)
    true_failure:  int             = 0
