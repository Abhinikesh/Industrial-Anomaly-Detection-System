from pymongo import MongoClient, DESCENDING, ASCENDING
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/anomaly_detection")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client.get_default_database()

# Collections
sensor_readings = db["sensor_readings"]
anomalies = db["anomalies"]

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
except Exception as e:
    print(f"[database] Note on index creation: {e}")
