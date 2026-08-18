# Model Evaluation Summary & Comparative Analysis

**Generated:** 2026-08-17 17:22:48 UTC  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 synthetic sensor records)  
**Evaluated Live Stream Sample:** 705 telemetry packets ingested into MongoDB (17 actual failure events)

---

## 1. Executive Performance Comparison

| Evaluation Metric | 🌲 Isolation Forest | 🧠 Deep Autoencoder | Delta / Winning Model |
|---|:---:|:---:|:---:|
| **Training Offline F1** | `0.168` | **`0.285`** | **Autoencoder (+69.6%)** |
| **Live Stream Precision** | `0.545` | `0.524` | Balanced (~52.4%) |
| **Live Stream Recall** | `0.353` | **`0.647`** | **Autoencoder (+29.4%)** |
| **Live Stream F1-Score** | `0.428` | **`0.579`** | **Autoencoder (Higher overall balance)** |
| **True Positives (TP)** | `6` | `11` | AE caught more true physical failures |
| **False Positives (FP)** | `5` | `10` | Acceptable false alarm rate |
| **False Negatives (FN)** | `11` | `6` | AE has fewer catastrophic misses |
| **Telemetry Trigger Rate** | `1.56%` (11 events) | `2.98%` (21 events) | Aligned with ~3.4% ground-truth failure rate |

---

## 2. Model Operational Mechanics & Technical Rationale

### 🌲 Model 1: Isolation Forest (Tree-Based Partitioning)
- **Algorithm Mechanism:** Operates by recursively partitioning continuous 5D sensor feature space with random axis-aligned orthogonal cuts across 100 decision trees (`n_estimators=100`). Anomalies require noticeably fewer random splits to isolate and end up near tree roots.
- **Contamination Prior:** Set explicitly to `0.034` (3.4%), mirroring the natural class imbalance discovered during initial Exploratory Data Analysis (EDA).
- **Strengths:** Extremely fast inference (<1ms), lightweight memory footprint, highly effective for sudden extreme point outliers in isolated single dimensions (e.g. abrupt torque spikes).
- **Weaknesses:** Cannot learn non-linear coupled relationships across multiple sensor axes (e.g. the inverse relationship where rotational speed and torque fluctuate simultaneously under normal motor load).

### 🧠 Model 2: Deep Autoencoder (PyTorch Neural Reconstruction)
- **Architecture:** Feedforward multi-layer bottleneck neural network (`5 → 3 → 2 → 3 → 5`) trained strictly on nominal machine operations (`Machine failure = 0`).
- **Anomaly Scoring Metric:** Row-wise Mean Squared Error (MSE) between original normalized sensor input and reconstructed vector:
  $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{5} (x_i - \hat{x}_i)^2$$
- **Decision Threshold:** Statistically calibrated to:
  $$\text{Threshold} = \mu_{\text{normal}} + 2\sigma_{\text{normal}} = 0.62011$$
- **Strengths:** Because the 2-neuron bottleneck compresses the manifold of nominal machine physics, any reading that violates physical sensor interdependencies produces high reconstruction error. This enables the model to catch complex multi-variable failure modes (such as Heat Dissipation Failure and Overstrain Failure).
- **Outcome:** Substantially higher recall (64.7% vs 35.3%), critical in industrial equipment where missing a failure results in costly unplanned machine downtime.

---

## 3. Real-Time Streaming Consensus & Divergence Analysis

During live telemetry streaming from the sensor simulator, model agreement statistics in MongoDB indicate:

- **Overall Agreement Rate:** **`97.73%`** (689 of 705 readings)
- **Consensus Normal Readings:** `681` readings (both models agreed system state was healthy)
- **Consensus Anomalies (High-Confidence):** `8` readings (both models concurrently triggered alert flags)
- **Autoencoder-Only Detections:** `13` readings (caught multi-sensor correlation drift that Isolation Forest missed)
- **Isolation Forest-Only Detections:** `3` readings (caught sharp 1D point spikes)

---

## 4. Production Ensemble Recommendation

For industrial condition monitoring, a **Union Ensemble (`is_anomaly = iso_flag OR ae_flag`)** is recommended and implemented in our FastAPI backend:
1. **Safety-First Posture:** Prioritizes high recall over precision; investigating a false alarm is negligible in cost compared to catastrophic spindle or tool failure.
2. **Complementary Coverage:** Combines the Autoencoder's relational sensitivity with the Isolation Forest's rapid point-outlier response.
