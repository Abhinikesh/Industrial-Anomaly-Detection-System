# Industrial Anomaly Detection System

Monitors simulated industrial sensor data (temperature, vibration, pressure, torque, tool wear) in real time, uses Isolation Forest and Autoencoder models to detect abnormal machine behaviour, and displays live charts and anomaly alerts on a React dashboard.

Built with a FastAPI backend and MongoDB for storage.

---

## What it does

- Generates continuous sensor readings from a simulated machine
- Runs two anomaly detection models (Isolation Forest + Autoencoder) on each reading
- Stores results in MongoDB
- Shows a live dashboard with sensor charts and flagged anomalies

---

## Project structure

```
industrial-anomaly-detection/
  backend/
    app/
      main.py          # FastAPI app entry point
      database.py      # MongoDB connection
      models/          # Pydantic request/response schemas
      routes/          # API route handlers
      services/        # Business logic (detection, storage)
    ml/
      simulator.py     # Generates fake sensor readings
      train_models.py  # Trains and saves Isolation Forest + Autoencoder
    requirements.txt
  frontend/            # React + Vite dashboard
  data/                # Raw or processed CSVs (optional)
  models/              # Saved model files after training
  results/             # Evaluation plots
  report_assets/       # Figures for the project report
```

---

## Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env              # edit MONGO_URI if needed

# train the models first (takes ~1–2 min)
cd ml
python train_models.py

# start the API
cd ..
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev                       # runs on http://localhost:3000
```

Make sure MongoDB is running locally before starting the backend.

---

## Running both servers

| Service  | URL                    |
|----------|------------------------|
| FastAPI  | http://localhost:8000  |
| React    | http://localhost:3000  |
| API docs | http://localhost:8000/docs |

---

## Models

- **Isolation Forest** — fast, good at global outliers
- **Autoencoder** — learns normal patterns, flags readings with high reconstruction error

Both are trained on simulated normal data with ~5% injected anomalies.

---

## Tech stack

- Python 3.10+, FastAPI, scikit-learn, TensorFlow, pymongo
- React 18, Vite, Recharts, Axios
- MongoDB (local)
# ndustrial-Anomaly-Detection-System
