from pydantic import BaseModel
from typing import Optional

class SensorReading(BaseModel):
    timestamp: str
    temperature: float
    vibration: float
    pressure: float
    torque: float
    tool_wear: float
    machine_id: Optional[str] = "machine_001"

class AnomalyRecord(BaseModel):
    timestamp: str
    machine_id: str
    anomaly_score: float
    is_anomaly: bool
    sensor_values: dict
