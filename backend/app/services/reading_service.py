from datetime import datetime, timezone
from app.database import sensor_readings, anomalies


def save_reading(doc: dict):
    """Insert one fully-scored reading into MongoDB."""
    sensor_readings.insert_one(doc)


def get_recent_readings(limit: int = 100) -> list:
    """Fetch the most recent N readings for the live chart, newest first."""
    cursor = (
        sensor_readings
        .find({}, {"_id": 0})        # exclude Mongo _id from response
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_anomalies(limit: int = 200) -> list:
    """Fetch readings flagged as anomalies, most recent first."""
    cursor = (
        sensor_readings
        .find({"is_anomaly": True}, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_reading_stats() -> dict:
    """Summary counts for the dashboard header cards."""
    total     = sensor_readings.count_documents({})
    anomaly_n = sensor_readings.count_documents({"is_anomaly": True})
    iso_only  = sensor_readings.count_documents({"iso_flag": 1, "ae_flag":  0})
    ae_only   = sensor_readings.count_documents({"iso_flag": 0, "ae_flag":  1})
    both      = sensor_readings.count_documents({"iso_flag": 1, "ae_flag":  1})

    return {
        "total_readings":  total,
        "total_anomalies": anomaly_n,
        "anomaly_rate":    round(anomaly_n / total * 100, 2) if total else 0.0,
        "flagged_by": {
            "isolation_forest_only": iso_only,
            "autoencoder_only":      ae_only,
            "both_models":           both,
        },
    }


def _calc_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = round(tp / (tp + fp), 3) if (tp + fp) > 0 else 0.0
    recall    = round(tp / (tp + fn), 3) if (tp + fn) > 0 else 0.0
    f1        = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "tn":        tn,
    }


def get_model_comparison_stats() -> dict:
    """
    Computes live streaming comparison metrics for Isolation Forest vs Autoencoder
    evaluated against true_failure on all stored readings in MongoDB.
    Safely handles empty database state (0 readings) without throwing division errors.
    """
    total = sensor_readings.count_documents({})
    if total == 0:
        empty_metrics = {
            "flagged": 0, "flagged_pct": 0.0,
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0
        }
        return {
            "total_readings": 0,
            "true_failures":  0,
            "true_normals":   0,
            "isolation_forest": empty_metrics,
            "autoencoder":      empty_metrics,
            "agreement": {
                "agreed_count":     0,
                "agreement_pct":    100.0,
                "disagreed_count":  0,
                "disagreement_pct": 0.0,
                "both_flagged":     0,
                "both_normal":      0,
                "iso_only":         0,
                "ae_only":          0,
            },
        }

    # True condition counts
    true_fails = sensor_readings.count_documents({"true_failure": 1})
    true_norms = total - true_fails

    # Isolation Forest metrics
    iso_flagged = sensor_readings.count_documents({"iso_flag": 1})
    tp_iso = sensor_readings.count_documents({"iso_flag": 1, "true_failure": 1})
    fp_iso = sensor_readings.count_documents({"iso_flag": 1, "true_failure": 0})
    fn_iso = sensor_readings.count_documents({"iso_flag": 0, "true_failure": 1})
    tn_iso = sensor_readings.count_documents({"iso_flag": 0, "true_failure": 0})
    iso_metrics = _calc_metrics(tp_iso, fp_iso, fn_iso, tn_iso)
    iso_metrics["flagged"] = iso_flagged
    iso_metrics["flagged_pct"] = round(iso_flagged / total * 100, 2)

    # Autoencoder metrics
    ae_flagged = sensor_readings.count_documents({"ae_flag": 1})
    tp_ae = sensor_readings.count_documents({"ae_flag": 1, "true_failure": 1})
    fp_ae = sensor_readings.count_documents({"ae_flag": 1, "true_failure": 0})
    fn_ae = sensor_readings.count_documents({"ae_flag": 0, "true_failure": 1})
    tn_ae = sensor_readings.count_documents({"ae_flag": 0, "true_failure": 0})
    ae_metrics = _calc_metrics(tp_ae, fp_ae, fn_ae, tn_ae)
    ae_metrics["flagged"] = ae_flagged
    ae_metrics["flagged_pct"] = round(ae_flagged / total * 100, 2)

    # Agreement / Disagreement
    both_flagged = sensor_readings.count_documents({"iso_flag": 1, "ae_flag": 1})
    both_normal  = sensor_readings.count_documents({"iso_flag": 0, "ae_flag": 0})
    iso_only     = sensor_readings.count_documents({"iso_flag": 1, "ae_flag": 0})
    ae_only      = sensor_readings.count_documents({"iso_flag": 0, "ae_flag": 1})

    agreed_count    = both_flagged + both_normal
    disagreed_count = iso_only + ae_only
    agreement_pct   = round(agreed_count / total * 100, 2)
    disagreement_pct= round(disagreed_count / total * 100, 2)

    return {
        "total_readings": total,
        "true_failures":  true_fails,
        "true_normals":   true_norms,
        "isolation_forest": iso_metrics,
        "autoencoder":      ae_metrics,
        "agreement": {
            "agreed_count":     agreed_count,
            "agreement_pct":    agreement_pct,
            "disagreed_count":  disagreed_count,
            "disagreement_pct": disagreement_pct,
            "both_flagged":     both_flagged,
            "both_normal":      both_normal,
            "iso_only":         iso_only,
            "ae_only":          ae_only,
        },
    }
