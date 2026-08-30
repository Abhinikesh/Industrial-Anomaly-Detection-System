# Industrial Anomaly Detection System

Real-time predictive maintenance platform for industrial machinery. Ingests continuous sensor telemetry from three independent machine fleets, scores every reading with two unsupervised ML models (**Isolation Forest** + **PyTorch Autoencoder**), stores results in **MongoDB**, and serves a live **React** mission-control dashboard with per-fleet filtering, anomaly incident logs, and side-by-side model benchmarks.

---

## Supported Machine Types

| Machine Type | Dataset | Features | Model Architecture | Failure Rate |
|---|---|---|---|---|
| `milling_machine` | [AI4I 2020 (UCI)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) | `air_temp`, `process_temp`, `rpm`, `torque`, `tool_wear` | IF (100 trees) + AE `5→3→2→3→5` | 3.4% |
| `fleet_machine` | [Azure PdM (Kaggle)](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) | `voltage`, `rotation`, `pressure`, `vibration` | IF (100 trees) + AE `4→3→2→3→4` | 2.0% |
| `water_pump` | [Pump Sensor (Kaggle)](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) | Top-15 of 51 sensors by failure correlation | IF (150 trees) + AE `15→10→5→10→15` + BatchNorm | 6.6% |

Each fleet has its own independent model pair. Readings never cross pipelines — the API automatically routes each ingest request to the correct model based on `machine_type`.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Simulators (3 × Python)                                     │
│  simulator.py · simulator_azure.py · simulator_pump.py       │
│  (chronological replay at configurable cadence)              │
└──────────────────┬───────────────────────────────────────────┘
                   │  POST /ingest  (JSON telemetry)
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  FastAPI Backend  :8000                                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Input Validation (Pydantic)                         │     │
│  │ • Unknown machine_type → 422                        │     │
│  │ • Missing sensor keys  → 422                        │     │
│  │ • Non-finite values    → 422                        │     │
│  └───────────────────┬─────────────────────────────────┘     │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ Model Registry (anomaly_service.py)                 │     │
│  │ machine_type → (IsolationForest, Scaler, Autoenc.)  │     │
│  │ All models loaded at startup, scored per-request    │     │
│  └───────────────────┬─────────────────────────────────┘     │
│                      ▼                                       │
│           iso_flag · ae_flag · is_anomaly                    │
│                      ▼                                       │
│  ┌──────────────────────────────┐                           │
│  │  MongoDB  (sensor_readings)  │◄── GET /readings/*        │
│  └──────────────────────────────┘                           │
└──────────────────────────────────────────────────────────────┘
                   │  REST (Axios)
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  React Dashboard  :3000                                      │
│  • Fleet selector  (All / Milling / Azure / Pump)            │
│  • Live sensor charts + anomaly incident log                 │
│  • Model comparison cards with per-fleet F1 / recall        │
│  • Training artefact gallery (confusion matrices, etc.)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

- **Backend & ML:** Python 3.10+, FastAPI, Uvicorn, PyTorch 2.x, Scikit-Learn, Pandas, NumPy, Joblib
- **Configuration:** `pydantic-settings` (`.env`-driven, all settings in `backend/app/config.py`)
- **Logging:** Python `logging` module — rotating file `logs/app.log` + stdout
- **Database:** MongoDB Community Server (local, `mongodb://localhost:27017`)
- **Frontend:** React 18, Vite, Recharts, Axios, Vanilla CSS (dark control-room theme)

---

## Prerequisites

- **Python 3.10+** — `python3 --version`
- **Node.js 18+** — `node -v`
- **MongoDB Community** running on port 27017 — `brew services start mongodb-community`
- **Kaggle API credentials** (for Azure PdM and Pump Sensor datasets):
  1. Go to [kaggle.com → Account → API → Create Token](https://www.kaggle.com/settings)
  2. Place `kaggle.json` at `~/.kaggle/kaggle.json`
  3. `chmod 600 ~/.kaggle/kaggle.json`

---

## Setup (fresh install)

### 1. Clone & configure environment
```bash
git clone https://github.com/Abhinikesh/Industrial-Anomaly-Detection-System.git
cd "Industrial Anomaly Detection System"

cp .env.example .env          # edit MONGO_URI if needed
```

### 2. Install Python dependencies
```bash
cd backend
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download datasets & train all three model pairs

> Run all commands from inside `backend/` with the venv active.

**Milling Machine (AI4I 2020 — auto-download from UCI, no Kaggle needed)**
```bash
python ml/download_dataset.py           # → data/raw/ai4i2020.csv
python ml/explore_data.py               # → results/eda/  (EDA plots)
python ml/train_isolation_forest.py     # → models/isolation_forest.pkl
python ml/train_autoencoder.py          # → models/autoencoder.pt
```

**Azure Fleet (Microsoft Azure PdM — requires Kaggle credentials)**
```bash
python ml/datasets/azure_pdm/download_dataset.py   # → data/raw/azure_pdm/
python ml/datasets/azure_pdm/preprocess.py         # → data/processed/azure_pdm.csv
python ml/train_azure_models.py                    # → models/azure/  (~3–5 min)
```

**Water Pump (Pump Sensor Data — requires Kaggle credentials)**
```bash
python ml/datasets/pump_sensor/download_dataset.py # → data/raw/pump_sensor/
python ml/datasets/pump_sensor/preprocess.py       # → data/processed/pump_sensor.csv
python ml/train_pump_models.py                     # → models/pump/  (~2–4 min)
```

### 4. Install frontend dependencies
```bash
cd ../frontend
npm install
cd ..
```

---

## Running the System

### Option A — Single command (recommended)
```bash
# From project root:
./start.sh
```
Starts MongoDB check → backend → frontend → all 3 simulators. Press `Ctrl+C` to stop everything.

### Option B — Manual (5 terminals)

**Terminal 1 — Backend**
```bash
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

**Terminal 2 — Frontend**
```bash
cd frontend && npm run dev
```
Dashboard: http://localhost:3000

**Terminal 3 — Milling Machine simulator**
```bash
cd backend && source venv/bin/activate
python ml/simulator.py --fast
```

**Terminal 4 — Azure Fleet simulator**
```bash
cd backend && source venv/bin/activate
python ml/simulator_azure.py --fast
```

**Terminal 5 — Water Pump simulator**
```bash
cd backend && source venv/bin/activate
python ml/simulator_pump.py --fast
```

---

## Project Structure

```
Industrial Anomaly Detection System/
├── .env.example                         # Environment template (copy to .env)
├── .env                                 # Local config (gitignored)
├── start.sh                             # All-in-one startup script
│
├── backend/
│   ├── app/
│   │   ├── config.py                    # Centralised settings (pydantic-settings)
│   │   ├── logger.py                    # Rotating file + console logger
│   │   ├── database.py                  # MongoDB connection & index setup
│   │   ├── main.py                      # FastAPI app entry point & CORS
│   │   ├── models/
│   │   │   └── reading.py               # Pydantic schemas + per-type validation
│   │   ├── routes/
│   │   │   ├── ingest.py                # POST /ingest
│   │   │   └── readings.py              # GET /readings/{recent,anomalies,stats,...}
│   │   └── services/
│   │       ├── anomaly_service.py       # Model registry + IF & AE scoring
│   │       └── reading_service.py       # MongoDB CRUD & benchmark aggregations
│   ├── ml/
│   │   ├── download_dataset.py          # AI4I 2020 downloader (UCI)
│   │   ├── explore_data.py              # EDA plots for milling machine
│   │   ├── train_isolation_forest.py    # Milling IF training
│   │   ├── train_autoencoder.py         # Milling AE training
│   │   ├── train_azure_models.py        # Azure fleet IF + AE training
│   │   ├── train_pump_models.py         # Water pump IF + AE training
│   │   ├── simulator.py                 # Milling machine telemetry streamer
│   │   ├── simulator_azure.py           # Azure fleet telemetry streamer
│   │   ├── simulator_pump.py            # Water pump telemetry streamer
│   │   └── datasets/
│   │       ├── azure_pdm/
│   │       │   ├── download_dataset.py  # Kaggle downloader
│   │       │   └── preprocess.py        # 24h look-ahead failure labelling
│   │       └── pump_sensor/
│   │           ├── download_dataset.py  # Kaggle downloader
│   │           └── preprocess.py        # Top-15 sensor selection by correlation
│   ├── logs/
│   │   └── app.log                      # Rotating application log (gitignored)
│   └── requirements.txt
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── LiveDashboard.jsx         # Live charts + fleet selector + incident log
│       │   └── ModelComparison.jsx       # Per-fleet IF vs AE benchmark cards
│       ├── App.jsx                       # Navigation tabs
│       └── App.css                       # Dark control-room theme
│
├── data/
│   ├── raw/                              # Downloaded raw CSVs (gitignored if large)
│   └── processed/                        # Preprocessed & labelled CSVs
│
├── models/
│   ├── isolation_forest.pkl              # Milling machine IF
│   ├── scaler.pkl                        # Milling machine scaler
│   ├── autoencoder.pt                    # Milling machine AE
│   ├── autoencoder_threshold.json        # Milling machine AE threshold
│   ├── azure/                            # Azure fleet models
│   └── pump/                             # Water pump models
│
└── results/
    ├── eda/                              # Milling machine EDA plots
    ├── isolation_forest/                 # Milling IF evaluation plots
    ├── autoencoder/                      # Milling AE evaluation plots
    ├── azure_pdm/                        # Azure fleet evaluation plots
    └── pump_sensor/                      # Water pump evaluation plots
```

---

## Configuration

All settings live in `backend/app/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017/anomaly_detection` | MongoDB connection string |
| `API_PORT` | `8000` | Uvicorn bind port |
| `CORS_ORIGINS` | `*` | Allowed origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `LOG_FILE` | `logs/app.log` | Rotating log file path (relative to `backend/`) |
| `LOG_MAX_BYTES` | `10485760` | Max log file size before rotation (10 MB) |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files to keep |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Score and store one sensor reading |
| `GET` | `/readings/recent` | Latest N readings (`?machine_type=`, `?limit=`) |
| `GET` | `/readings/anomalies` | Latest N anomalous readings (`?machine_type=`) |
| `GET` | `/readings/stats` | Aggregate counts and anomaly rate (`?machine_type=`) |
| `GET` | `/readings/model-comparison` | IF vs AE benchmark metrics (`?machine_type=`) |
| `GET` | `/readings/fleet-overview` | One-row summary per machine type |
| `GET` | `/health` | Liveness probe |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Known Limitations

- **Chronological replay only:** Simulators replay historical datasets at a configurable cadence. Live hardware sensor protocols (MQTT / Modbus / OPC-UA) are not yet integrated.
- **Union ensemble:** The `is_anomaly` flag uses an OR rule (flagged if *either* model fires) for maximum recall. Weighted voting or cost-sensitive thresholds are not yet implemented.
- **Single-node deployment:** MongoDB runs locally. Production hardening (replica set, TLS, auth) is out of scope for this version.
