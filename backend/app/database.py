from pymongo import MongoClient, DESCENDING
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
    sensor_readings.create_index([("timestamp", DESCENDING)])
    sensor_readings.create_index([("is_anomaly", DESCENDING), ("timestamp", DESCENDING)])
except Exception as e:
    print(f"[database] Note on index creation: {e}")
