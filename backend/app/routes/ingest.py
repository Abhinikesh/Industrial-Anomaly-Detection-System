from fastapi import APIRouter, HTTPException
from app.models.reading import IngestPayload
from app.services.anomaly_service import score_reading
from app.services.reading_service import save_reading

router = APIRouter()


@router.post("/ingest")
def ingest(payload: IngestPayload):
    """
    Receives one sensor reading from the simulator, scores it with both models,
    persists the full result to MongoDB, and returns the scoring output so the
    simulator can log what happened.
    """
    scores = score_reading(payload.model_dump())

    doc = {
        **payload.model_dump(),
        **scores,
    }

    try:
        save_reading(doc)
    except Exception as e:
        # don't let a DB write failure kill the scoring response
        print(f"[ingest] MongoDB write failed: {e}")

    return {
        "status":      "ok",
        "timestamp":   payload.timestamp,
        "machine_id":  payload.machine_id,
        "iso_score":   scores["iso_score"],
        "iso_flag":    scores["iso_flag"],
        "ae_score":    scores["ae_score"],
        "ae_flag":     scores["ae_flag"],
        "is_anomaly":  scores["is_anomaly"],
        "true_failure": payload.true_failure,
    }
