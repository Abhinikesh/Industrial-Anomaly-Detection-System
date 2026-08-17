from datetime import datetime, timezone
from app.database import sensor_readings, anomalies


def save_reading(doc: dict):
    """Insert one fully-scored reading into MongoDB."""
    sensor_readings.insert_one(doc)


def get_recent_readings(limit: int = 100) -> list:
    """Fetch the most recent N readings for the live chart, newest first."""
    cursor = (
        sensor_readings
        .find({}, {"_id": 0})        # exclude Mongo _id from response
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_anomalies(limit: int = 200) -> list:
    """Fetch readings flagged as anomalies, most recent first."""
    cursor = (
        sensor_readings
        .find({"is_anomaly": True}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_reading_stats() -> dict:
    """Summary counts for the dashboard header cards."""
    total     = sensor_readings.count_documents({})
    anomaly_n = sensor_readings.count_documents({"is_anomaly": True})
    iso_only  = sensor_readings.count_documents({"iso_flag": 1, "ae_flag":  0})
    ae_only   = sensor_readings.count_documents({"iso_flag": 0, "ae_flag":  1})
    both      = sensor_readings.count_documents({"iso_flag": 1, "ae_flag":  1})

    return {
        "total_readings":  total,
        "total_anomalies": anomaly_n,
        "anomaly_rate":    round(anomaly_n / total * 100, 2) if total else 0,
        "flagged_by": {
            "isolation_forest_only": iso_only,
            "autoencoder_only":      ae_only,
            "both_models":           both,
        },
    }
