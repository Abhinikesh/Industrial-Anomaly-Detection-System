from pymongo import MongoClient, DESCENDING, ASCENDING

from app.config import settings
from app.logger import get_logger

log = get_logger(__name__)

client = MongoClient(
    settings.mongo_uri,
    serverSelectionTimeoutMS=settings.mongo_server_selection_timeout_ms,
)
db = client.get_default_database()

# Collections
sensor_readings = db["sensor_readings"]
anomalies       = db["anomalies"]

# Ensure performance indexes for high-throughput streaming and fast dashboard queries
try:
    # Primary sort index — most dashboard queries use this
    sensor_readings.create_index([("timestamp", DESCENDING)])

    # Anomaly filter + sort — used by /readings/anomalies
    sensor_readings.create_index(
        [("is_anomaly", DESCENDING), ("timestamp", DESCENDING)]
    )

    # Machine-type filter + sort — enables per-fleet queries without full scans
    sensor_readings.create_index(
        [("machine_type", ASCENDING), ("timestamp", DESCENDING)]
    )

    # Machine-type + anomaly compound — for per-type anomaly filtered queries
    sensor_readings.create_index(
        [("machine_type", ASCENDING), ("is_anomaly", DESCENDING), ("timestamp", DESCENDING)]
    )

    log.info("MongoDB indexes verified/created on sensor_readings collection.")
except Exception as e:
    log.warning("Index creation note (non-fatal): %s", e)
