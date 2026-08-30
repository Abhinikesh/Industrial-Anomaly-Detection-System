from fastapi import APIRouter, HTTPException
from app.models.reading import IngestPayload
from app.services.anomaly_service import score_reading
from app.services.reading_service import save_reading

router = APIRouter()


@router.post("/ingest")
def ingest(payload: IngestPayload):
    """
    Receives one sensor reading from a simulator or edge client, scores it
    with the model pair registered for payload.machine_type, persists the
    fully-scored document to MongoDB, and returns the scoring output.
    """
    scores = score_reading(payload.model_dump())

    doc = {
        **payload.model_dump(),
        **scores,
    }

    try:
        save_reading(doc)
    except Exception as e:
        # Don't let a DB write failure kill the scoring response
        print(f"[ingest] MongoDB write failed: {e}")

    return {
        "status":        "ok",
        "timestamp":     payload.timestamp,
        "machine_type":  payload.machine_type,
        "machine_id":    payload.machine_id,
        "iso_score":     scores["iso_score"],
        "iso_flag":      scores["iso_flag"],
        "ae_score":      scores["ae_score"],
        "ae_flag":       scores["ae_flag"],
        "is_anomaly":    scores["is_anomaly"],
        "true_failure":  payload.true_failure,
    }
