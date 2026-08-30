from typing import Optional, List
from app.database import sensor_readings


def save_reading(doc: dict):
    """Insert one fully-scored reading into MongoDB."""
    sensor_readings.insert_one(doc)


# ── Query helpers ─────────────────────────────────────────────────────────────

def _type_filter(machine_type: Optional[str]) -> dict:
    """Return a MongoDB filter dict scoped to a machine type (or empty = all)."""
    return {"machine_type": machine_type} if machine_type else {}


def get_recent_readings(limit: int = 100, machine_type: Optional[str] = None) -> list:
    """Fetch the most recent N readings for the live chart, newest first.

    Pass machine_type to scope results to a single fleet type (e.g. 'milling_machine').
    Omit (or pass None) to return readings across all machine types.
    """
    cursor = (
        sensor_readings
        .find(_type_filter(machine_type), {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_anomalies(limit: int = 200, machine_type: Optional[str] = None) -> list:
    """Fetch readings flagged as anomalies, most recent first.

    Optionally scoped to one machine_type.
    """
    filt = {**_type_filter(machine_type), "is_anomaly": True}
    cursor = (
        sensor_readings
        .find(filt, {"_id": 0})
        .sort("timestamp", -1)
        .limit(limit)
    )
    return list(cursor)


def get_reading_stats(machine_type: Optional[str] = None) -> dict:
    """Summary counts for the dashboard header cards.

    When machine_type is provided, all counts are scoped to that type.
    """
    base = _type_filter(machine_type)

    total     = sensor_readings.count_documents(base)
    anomaly_n = sensor_readings.count_documents({**base, "is_anomaly": True})
    iso_only  = sensor_readings.count_documents({**base, "iso_flag": 1, "ae_flag": 0})
    ae_only   = sensor_readings.count_documents({**base, "iso_flag": 0, "ae_flag": 1})
    both      = sensor_readings.count_documents({**base, "iso_flag": 1, "ae_flag": 1})

    return {
        "total_readings":  total,
        "total_anomalies": anomaly_n,
        "anomaly_rate":    round(anomaly_n / total * 100, 2) if total else 0.0,
        "machine_type":    machine_type or "all",
        "flagged_by": {
            "isolation_forest_only": iso_only,
            "autoencoder_only":      ae_only,
            "both_models":           both,
        },
    }


# ── Metric helpers ────────────────────────────────────────────────────────────

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


def get_fleet_overview() -> List[dict]:
    """Return one summary row per machine_type for the 'All' fleet overview panel."""
    pipeline = [
        {
            "$group": {
                "_id":         {"$ifNull": ["$machine_type", "milling_machine"]},
                "total":       {"$sum": 1},
                "anomalies":   {"$sum": {"$cond": ["$is_anomaly", 1, 0]}},
                "latest_ts":   {"$max": "$timestamp"},
                "latest_anom": {"$last": "$is_anomaly"},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    rows = list(sensor_readings.aggregate(pipeline))
    result = []
    for r in rows:
        total = r["total"] or 1
        result.append({
            "machine_type":    r["_id"],
            "total_readings":  r["total"],
            "anomaly_count":   r["anomalies"],
            "anomaly_rate":    round(r["anomalies"] / total * 100, 2),
            "latest_ts":       r["latest_ts"],
            "is_anomaly_now":  bool(r["latest_anom"]),
        })
    return result


def get_model_comparison_stats(machine_type: Optional[str] = None) -> dict:
    """
    Computes live streaming comparison metrics for Isolation Forest vs Autoencoder
    evaluated against true_failure on stored readings in MongoDB.
    Safely handles empty database state (0 readings) without throwing division errors.

    Pass machine_type to scope metrics to one fleet type. Omit for all types combined.
    """
    base = _type_filter(machine_type)
    total = sensor_readings.count_documents(base)
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
    true_fails = sensor_readings.count_documents({**base, "true_failure": 1})
    true_norms = total - true_fails

    # Isolation Forest metrics
    iso_flagged = sensor_readings.count_documents({**base, "iso_flag": 1})
    tp_iso = sensor_readings.count_documents({**base, "iso_flag": 1, "true_failure": 1})
    fp_iso = sensor_readings.count_documents({**base, "iso_flag": 1, "true_failure": 0})
    fn_iso = sensor_readings.count_documents({**base, "iso_flag": 0, "true_failure": 1})
    tn_iso = sensor_readings.count_documents({**base, "iso_flag": 0, "true_failure": 0})
    iso_metrics = _calc_metrics(tp_iso, fp_iso, fn_iso, tn_iso)
    iso_metrics["flagged"]     = iso_flagged
    iso_metrics["flagged_pct"] = round(iso_flagged / total * 100, 2)

    # Autoencoder metrics
    ae_flagged = sensor_readings.count_documents({**base, "ae_flag": 1})
    tp_ae = sensor_readings.count_documents({**base, "ae_flag": 1, "true_failure": 1})
    fp_ae = sensor_readings.count_documents({**base, "ae_flag": 1, "true_failure": 0})
    fn_ae = sensor_readings.count_documents({**base, "ae_flag": 0, "true_failure": 1})
    tn_ae = sensor_readings.count_documents({**base, "ae_flag": 0, "true_failure": 0})
    ae_metrics = _calc_metrics(tp_ae, fp_ae, fn_ae, tn_ae)
    ae_metrics["flagged"]     = ae_flagged
    ae_metrics["flagged_pct"] = round(ae_flagged / total * 100, 2)

    # Agreement / Disagreement
    both_flagged = sensor_readings.count_documents({**base, "iso_flag": 1, "ae_flag": 1})
    both_normal  = sensor_readings.count_documents({**base, "iso_flag": 0, "ae_flag": 0})
    iso_only     = sensor_readings.count_documents({**base, "iso_flag": 1, "ae_flag": 0})
    ae_only      = sensor_readings.count_documents({**base, "iso_flag": 0, "ae_flag": 1})

    agreed_count     = both_flagged + both_normal
    disagreed_count  = iso_only + ae_only
    agreement_pct    = round(agreed_count / total * 100, 2)
    disagreement_pct = round(disagreed_count / total * 100, 2)

    return {
        "machine_type": machine_type or "all",
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
