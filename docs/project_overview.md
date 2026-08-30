# Industrial Anomaly Detection System — Project Overview

**Project Repository:** `industrial-anomaly-detection`  
**Domain:** Predictive Maintenance (PdM) & Industrial Internet of Things (IIoT)  
**Authorship / Role:** End-to-End Machine Learning & Full-Stack Implementation

---

## 1. Problem Statement

Unplanned equipment downtime represents one of the single largest sources of financial loss in modern industrial manufacturing, costing global manufacturing sectors hundreds of billions of dollars annually. Catastrophic machine failures (e.g. spindle seizure, tool breakages, motor overheat, overstrain) lead to:
1. **Excessive Maintenance Costs:** Emergency repairs cost up to $3\times$ to $5\times$ more than proactive condition-based servicing.
2. **Production Bottlenecks:** A single bottleneck machine breakdown halts entire assembly lines.
3. **Safety Hazards:** Severe overstrain or structural failure of rotating tooling poses acute physical risks to factory personnel.

Traditional preventative maintenance relies on static time-based servicing schedules, which either replace components prematurely (wasting operational lifespan) or service equipment too late. A real-time **Predictive Anomaly Detection System** continuously monitors telemetry to flag abnormal behavior *before* irreversible failure occurs.

---

## 2. Project Objectives

- **Unsupervised Telemetry Scoring:** Build and compare multiple machine learning algorithms capable of detecting anomalous physical states without requiring labeled failure data during training.
- **Dual-Model Comparative Study:** Benchmark traditional tree-based partitioning (**Isolation Forest**) against deep neural reconstruction (**PyTorch Autoencoder**) on realistic physical sensor streams.
- **End-to-End Real-Time Pipeline:** Construct an asynchronous backend API and database to ingest continuous sensor readings at 1-second cadence.
- **Interactive Mission-Control UI:** Deliver a high-contrast React dashboard displaying live trend lines, pulsing anomaly alerts, incident feeds, and dynamic model benchmark metrics.

---

## 3. Tech Stack & Architectural Justification

| Layer | Selected Technology | Architectural Justification |
|---|---|---|
| **Data & ML** | `Python 3.14`, `PyTorch 2.13`, `Scikit-Learn`, `Pandas`, `NumPy` | Python provides rich scientific tooling. PyTorch was chosen for the neural autoencoder to support seamless model definition, non-linear activations, and high numerical stability. |
| **Backend API** | `FastAPI`, `Uvicorn`, `Pydantic` | Asynchronous execution model allows high-throughput telemetry ingestion with minimal latency (<30ms round-trip). Built-in Pydantic validation guarantees schema safety. |
| **Storage Layer** | `MongoDB (Community Server)`, `PyMongo` | Flexible JSON document model matches sensor telemetry streams without rigid migration overhead. Descending B-tree indexes guarantee $O(\log N)$ retrieval for live charts. |
| **Frontend Dashboard** | `React 18`, `Vite`, `Recharts`, `Axios` | Single Page Application (SPA) architecture with Recharts enables smooth rendering of 50-point live sensor trend lines without UI freezing. |

### Why Compare Isolation Forest AND Autoencoder?
Rather than developing an isolated black-box model, this project implements a **comparative machine learning study**:
- **Isolation Forest** serves as the classic, fast, tree-based baseline to test if failures can be caught as simple orthogonal point outliers.
- **Deep Autoencoder** acts as the modern neural baseline to test if failures represent subtle multi-sensor correlation drift (e.g. RPM dropping while Torque spikes).
- Together, they demonstrate how different algorithmic paradigms perceive identical industrial telemetry.

---

## 4. Hardware Simulation Justification

In professional industrial machine learning development, accessing dedicated multi-million-dollar CNC milling machines or physical factory test rigs for early algorithmic R&D is often impractical and hazardous (intentionally driving physical machinery to destruction creates serious safety and capital risks).

Using the **AI4I 2020 Predictive Maintenance Dataset** (a rigorous simulation reflecting real milling machine physics according to ISO standards) combined with a real-time playback simulator:
- Generates realistic, continuous physical sensor streams (Torque, Tool Wear, Spindle RPM, Temperatures).
- Retains authentic failure modes: Heat Dissipation Failure (HDF), Overstrain Failure (OSF), Power Failure (PWF), and Tool Wear Failure (TWF).
- Replays data chronologically to mirror live IoT edge telemetry feeds.

---

## 5. Key Implemented Features

1. **Dataset Ingestion & EDA Pipeline (`explore_data.py`):**
   - Automated download and extraction from UCI Machine Learning Repository.
   - Comprehensive correlation heatmaps, class imbalance profiling ($3.4\%$ natural failure rate), and distribution plots.
2. **Isolation Forest Pipeline (`train_isolation_forest.py`):**
   - Feature scaling via `StandardScaler` to prevent RPM magnitudes from dominating tree splits.
   - Contamination calibration ($0.034$) and model artifact persistence via `joblib`.
3. **PyTorch Deep Autoencoder (`train_autoencoder.py`):**
   - $5 \to 3 \to 2 \to 3 \to 5$ bottleneck compression architecture trained strictly on nominal data.
   - Statistical anomaly thresholding ($\mu + 2\sigma = 0.62011$).
4. **Resilient Telemetry Simulator (`simulator.py`):**
   - Continuous cyclic streaming with adjustable cadence (`--fast` / `--delay`).
   - Connection retry backoff and auto-recovery logic when backend restarts.
5. **High-Performance FastAPI Ingest Engine (`app/main.py`):**
   - RESTful endpoints for telemetry ingestion, live chart queries, incident filtering, and model benchmark calculation.
   - Indexed MongoDB persistence.
6. **Mission-Control React Dashboard (`frontend/src/`):**
   - Real-time line charts for Thermal, Mechanical, and Tool Wear profiles with glowing anomaly dots.
   - Side-by-side model comparison view with live consensus progress bars and artifact galleries.

---

## 6. Engineering Challenges & Solutions

| # | Real Challenge Encountered | Root Cause | Engineering Solution Implemented |
|---|---|---|---|
| **1** | **Python 3.14 Environment Compatibility** | Initial setup attempted TensorFlow installation, but TensorFlow lacks Python 3.14 pre-compiled wheels. | Migrated the Autoencoder architecture to native **PyTorch 2.13**, which provides full compatibility with modern Python 3.14 runtimes on macOS/ARM. |
| **2** | **Autoencoder Anomaly Threshold Selection** | Setting threshold too low caused high false alarm rates; setting it too high missed true machine failures. | Calibrated the decision threshold statistically using the reconstruction error distribution of normal training data ($\mu + 2\sigma$). This established an empirical boundary capturing $97.7\%$ of normal variation. |
| **3** | **Ensemble Decision Logic (Union vs Intersection)** | Determining how to combine predictions from two models with different underlying score scales. | Adopted a **Union (OR) Rule** (`is_anomaly = iso_flag OR ae_flag`). In industrial equipment monitoring, safety and recall supersede precision — catching failures early outweighs investigating an occasional false alarm. |
| **4** | **Live Polling Latency as Database Grew** | Polling `/readings/recent` caused full collection scans ($O(N)$) as thousands of readings accumulated in MongoDB. | Enforced compound B-tree indexes (`[("timestamp", -1)]` and `[("is_anomaly", -1), ("timestamp", -1)]`) in `database.py` on startup, maintaining sub-millisecond query responses regardless of collection size. |

---

## 7. Future Scope & Industrial Extensions

1. **Temporal Sequence Models (LSTM Autoencoders):** Implement Recurrent / Long Short-Term Memory (LSTM) Autoencoders to analyze sliding time-windows ($t_{-10} \dots t_0$) for gradual mechanical drift rather than instantaneous point vectors.
2. **Multi-Machine Fleet Simulation:** Extend the simulator and database schema to ingest and differentiate telemetry across dozens of distributed machine nodes simultaneously.
3. **Industrial IoT Edge Protocols (MQTT / OPC-UA):** Replace HTTP POST polling with lightweight publish-subscribe protocols (MQTT / WebSockets / Kafka) to simulate industrial edge-gateway communications.
4. **Automated Continuous Retraining:** Build automated cron/trigger pipelines that periodically retrain the Autoencoder baseline on newly verified normal operational cycles stored in MongoDB.
