from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/anomaly_detection")

client = MongoClient(MONGO_URI)
db = client.get_default_database()

# Collections
sensor_readings = db["sensor_readings"]
anomalies = db["anomalies"]
