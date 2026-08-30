from typing import Optional
from fastapi import APIRouter
from app.services.reading_service import (
    get_recent_readings,
    get_anomalies,
    get_reading_stats,
    get_model_comparison_stats,
    get_fleet_overview,
)

router = APIRouter(prefix="/readings", tags=["readings"])


@router.get("/recent")
def recent_readings(
    limit: int = 100,
    machine_type: Optional[str] = None,
):
    """Return the most recent N readings.

    Optional query param:
      ?machine_type=milling_machine   — filter to one fleet type
      (omit to return readings across all machine types)
    """
    return get_recent_readings(limit=limit, machine_type=machine_type)


@router.get("/anomalies")
def anomaly_readings(
    limit: int = 200,
    machine_type: Optional[str] = None,
):
    """Return the most recent N anomalous readings.

    Optional query param:
      ?machine_type=milling_machine   — filter to one fleet type
    """
    return get_anomalies(limit=limit, machine_type=machine_type)


@router.get("/stats")
def reading_stats(machine_type: Optional[str] = None):
    """Return aggregate counts (total, anomaly rate, flagged-by breakdown).

    Optional query param:
      ?machine_type=milling_machine   — scope counts to one fleet type
    """
    return get_reading_stats(machine_type=machine_type)


@router.get("/model-comparison")
def model_comparison(machine_type: Optional[str] = None):
    """Return side-by-side IF vs AE benchmark metrics.

    Optional query param:
      ?machine_type=milling_machine  — scope to one fleet type
      (omit for combined across all types)
    """
    return get_model_comparison_stats(machine_type=machine_type)


@router.get("/fleet-overview")
def fleet_overview():
    """Return one summary row per machine_type for the 'All' fleet view."""
    return get_fleet_overview()
