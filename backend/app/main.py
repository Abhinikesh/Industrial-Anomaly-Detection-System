from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.ingest import router as ingest_router
from app.services.reading_service import (
    get_recent_readings,
    get_anomalies,
    get_reading_stats,
)

app = FastAPI(title="Industrial Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)


@app.get("/")
def root():
    return {"status": "Anomaly Detection API running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/readings/recent")
def recent_readings(limit: int = 100):
    return get_recent_readings(limit)


@app.get("/readings/anomalies")
def anomaly_readings(limit: int = 200):
    return get_anomalies(limit)


@app.get("/readings/stats")
def reading_stats():
    return get_reading_stats()
