# Industrial Anomaly Detection System

Real-time predictive maintenance for 3 machine fleets — Isolation Forest + PyTorch Autoencoder, FastAPI backend, React dashboard.

| Machine Type | Dataset | Sensors |
|---|---|---|
| `milling_machine` | AI4I 2020 (UCI) | air_temp, process_temp, rpm, torque, tool_wear |
| `fleet_machine` | Azure PdM (Kaggle) | voltage, rotation, pressure, vibration |
| `water_pump` | Pump Sensor (Kaggle) | top-15 of 51 sensors |

---

## Prerequisites

- Python 3.10+, Node.js 18+, MongoDB running on port 27017
- Kaggle credentials at `~/.kaggle/kaggle.json` (for Azure + Pump datasets)

---

## Setup

```bash
# 1. Python environment
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download & train — Milling Machine (no Kaggle needed)
python ml/download_dataset.py
python ml/train_isolation_forest.py && python ml/train_autoencoder.py

# 3. Download & train — Azure Fleet
python ml/datasets/azure_pdm/download_dataset.py
python ml/datasets/azure_pdm/preprocess.py
python ml/train_azure_models.py

# 4. Download & train — Water Pump
python ml/datasets/pump_sensor/download_dataset.py
python ml/datasets/pump_sensor/preprocess.py
python ml/train_pump_models.py

# 5. Frontend
cd ../frontend && npm install
```

---

## Run

### One command
```bash
./start.sh
```

### Or manually (5 terminals)
```bash
# Backend
cd backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm run dev

# Simulators (each in its own terminal)
cd backend && source venv/bin/activate && python ml/simulator.py --fast
cd backend && source venv/bin/activate && python ml/simulator_azure.py --fast
cd backend && source venv/bin/activate && python ml/simulator_pump.py --fast
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:3000 |
| API Docs | http://localhost:8000/docs |
| Logs | `backend/logs/app.log` |
