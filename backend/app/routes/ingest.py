from fastapi import APIRouter

from app.logger import get_logger
from app.models.reading import IngestPayload
from app.services.anomaly_service import score_reading
from app.services.reading_service import save_reading

log    = get_logger(__name__)
router = APIRouter()


@router.post("/ingest")
def ingest(payload: IngestPayload):
    """Receive one sensor reading, score it with the registered model pair
    for its machine_type, persist to MongoDB, and return scoring output.

    Pydantic validates the payload before this function is called:
    - Unknown machine_type  → 422 Unprocessable Entity
    - Missing sensor keys   → 422 Unprocessable Entity
    - Non-finite values     → 422 Unprocessable Entity
    """
    log.info(
        "Ingest: machine_type=%s machine_id=%s ts=%s",
        payload.machine_type,
        payload.machine_id,
        payload.timestamp,
    )

    scores = score_reading(payload.model_dump())

    # Warn if the scorer fell back to defaults (model missing / scoring error)
    if "model_warning" in scores:
        log.warning(
            "Score fallback for machine_type=%s: %s",
            payload.machine_type,
            scores["model_warning"],
        )

    doc = {**payload.model_dump(), **scores}

    try:
        save_reading(doc)
    except Exception as exc:
        # Don't let a DB write failure kill the scoring response
        log.error("MongoDB write failed for machine_id=%s: %s", payload.machine_id, exc)

    if scores.get("is_anomaly"):
        log.warning(
            "ANOMALY detected — machine_type=%s machine_id=%s iso=%.4f ae=%.4f",
            payload.machine_type,
            payload.machine_id,
            scores.get("iso_score", 0.0),
            scores.get("ae_score", 0.0),
        )

    return {
        "status":       "ok",
        "timestamp":    payload.timestamp,
        "machine_type": payload.machine_type,
        "machine_id":   payload.machine_id,
        "iso_score":    scores["iso_score"],
        "iso_flag":     scores["iso_flag"],
        "ae_score":     scores["ae_score"],
        "ae_flag":      scores["ae_flag"],
        "is_anomaly":   scores["is_anomaly"],
        "true_failure": payload.true_failure,
    }
