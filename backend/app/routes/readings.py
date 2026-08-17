from fastapi import APIRouter
from app.services.reading_service import (
    get_recent_readings,
    get_anomalies,
    get_reading_stats,
    get_model_comparison_stats,
)

router = APIRouter(prefix="/readings", tags=["readings"])


@router.get("/recent")
def recent_readings(limit: int = 100):
    return get_recent_readings(limit)


@router.get("/anomalies")
def anomaly_readings(limit: int = 200):
    return get_anomalies(limit)


@router.get("/stats")
def reading_stats():
    return get_reading_stats()


@router.get("/model-comparison")
def model_comparison():
    return get_model_comparison_stats()
