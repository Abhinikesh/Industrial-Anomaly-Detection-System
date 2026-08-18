# Industrial Anomaly Detection System

Real-time sensor monitoring and predictive maintenance system for industrial machinery. Continuously ingests physical telemetry (temperature, rotational speed, torque, tool wear), runs dual unsupervised anomaly detection models (**Isolation Forest** and a **PyTorch Autoencoder**) to detect abnormal operating states, stores scored records in **MongoDB**, and visualizes live telemetry streams and benchmark metrics on a **React** mission-control dashboard.

---

## Tech Stack

- **Backend & ML:** Python 3.10+, FastAPI, Uvicorn, PyTorch 2.x, Scikit-Learn, Pandas, NumPy, Joblib, PyMongo
- **Database:** MongoDB Community Server (local instance at `mongodb://localhost:27017`)
- **Frontend Dashboard:** React 18, Vite, Recharts, Axios, Vanilla CSS (Control-Room Dark Theme)
- **Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository, 10,000 records)

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:
- **Python 3.10+** (`python3 --version`)
- **Node.js 18+** and `npm` (`node -v`, `npm -v`)
- **MongoDB Community Edition** running locally on port 27017 (`mongosh` or `brew services start mongodb-community`)

---

## Step-by-Step Setup Guide

### 1. Clone & Setup Environment Files
```bash
git clone https://github.com/Abhinikesh/ndustrial-Anomaly-Detection-System.git
cd "Industrial Anomaly Detection System"

# Copy example environment configuration
cp .env.example .env
cp backend/.env.example backend/.env
```

### 2. Backend Setup & Dependency Installation
```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download Dataset & Train Baseline Models
*Note: Run these once before starting the servers to download the raw dataset and generate model weights in `models/`.*

```bash
# Still inside backend/ with venv active:
python ml/download_dataset.py       # Downloads & extracts data/raw/ai4i2020.csv
python ml/explore_data.py           # Generates EDA plots into results/eda/
python ml/train_isolation_forest.py # Trains IF, outputs models/isolation_forest.pkl
python ml/train_autoencoder.py      # Trains PyTorch AE, outputs models/autoencoder.pt
cd ..
```

### 4. Frontend Setup & Dependency Installation
```bash
cd frontend
npm install
cd ..
```

---

## Running the Application (4-Terminal Setup)

For demo day or development, open separate terminal windows and run in this order:

### Terminal 1: MongoDB Database
```bash
# Start MongoDB service if not already active
brew services start mongodb-community
# Or run mongod directly:
mongod --dbpath /usr/local/var/mongodb
```

### Terminal 2: FastAPI Backend Server
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
*API will be active at `http://localhost:8000` (Interactive docs at `http://localhost:8000/docs`).*

### Terminal 3: React Frontend Dashboard
```bash
cd frontend
npm run dev -- --port 3000
```
*Open your browser and navigate to **`http://localhost:3000`**.*

### Terminal 4: Sensor Stream Simulator
```bash
cd backend
source venv/bin/activate

# Standard 1-second cadence:
python ml/simulator.py

# Or fast test mode (0.2s cadence):
python ml/simulator.py --fast
```

---

## Quick Startup (All-in-One Script)

Alternatively, run the automated startup script:
```bash
./start.sh
```

---

## Project Structure

```
industrial-anomaly-detection/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   ├── reading.py             # Pydantic schema for telemetry & DB documents
│   │   │   └── schemas.py             # API schemas
│   │   ├── routes/
│   │   │   ├── ingest.py              # POST /ingest (telemetry receiver & scoring)
│   │   │   └── readings.py            # GET /readings/* (recent, stats, anomalies, comparison)
│   │   ├── services/
│   │   │   ├── anomaly_service.py     # Inference engine (loads IF + PyTorch AE)
│   │   │   └── reading_service.py     # MongoDB CRUD & live benchmark calculations
│   │   ├── database.py                # PyMongo connection & B-tree index setup
│   │   └── main.py                    # FastAPI application entry & CORS configuration
│   ├── ml/
│   │   ├── download_dataset.py        # Automated UCI dataset downloader
│   │   ├── explore_data.py            # EDA statistical script & histogram generator
│   │   ├── train_isolation_forest.py  # Isolation Forest training script
│   │   ├── train_autoencoder.py       # PyTorch Autoencoder training script
│   │   ├── simulator.py               # Continuous telemetry streaming engine
│   │   ├── generate_report_metrics.py # Markdown evaluation summary generator
│   │   └── generate_architecture_diagram.py # Architecture graphic generator
│   ├── requirements.txt               # Backend Python dependencies
│   ├── .env.example                   # Backend environment template
│   └── .env                           # Local environment config
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── LiveDashboard.jsx      # Real-time Recharts monitoring & incident log
│   │   │   └── ModelComparison.jsx    # Side-by-side benchmark metrics & image gallery
│   │   ├── App.jsx                    # Root view with navigation tabs
│   │   ├── App.css                    # Control-room styling & responsive grid layout
│   │   └── index.css                  # Global background reset
│   ├── package.json                   # React dependencies (Axios, Recharts, Vite)
│   └── vite.config.js                 # Dev server port 3000 configuration
├── data/
│   └── raw/                           # Raw AI4I 2020 CSV dataset (downloaded)
├── models/
│   ├── isolation_forest.pkl           # Serialized Isolation Forest model
│   ├── scaler.pkl                     # StandardScaler feature normalizer
│   ├── autoencoder.pt                 # PyTorch Autoencoder neural weights
│   └── autoencoder_threshold.json     # Calibrated reconstruction threshold (mean + 2σ)
├── results/
│   ├── eda/                           # Correlation heatmap and sensor distributions
│   ├── isolation_forest/              # IF confusion matrix & score distribution
│   ├── autoencoder/                   # AE confusion matrix & reconstruction error plots
│   └── screenshots/                   # Folder for live dashboard screenshots
├── report_assets/
│   ├── project_overview.md            # Comprehensive internship report overview
│   ├── model_metrics.md               # Auto-generated model benchmark comparison
│   ├── database_schema.md             # Collection schema & field documentation
│   ├── architecture_diagram.md        # Architecture narrative & Mermaid flowchart
│   ├── architecture_diagram.png       # Standalone system architecture diagram
│   └── sample_results/                # Curated evaluation plots for report insertion
├── start.sh                           # All-in-one startup script
├── .gitignore                         # Project-wide Git ignore rules
├── .env.example                       # Root environment template
└── README.md                          # Project documentation
```

---

## Known Limitations

- **Single-Machine Simulation:** The current edge simulator replays telemetry from one milling machine at a time (`machine_id: "SIM-001"`). Multi-tenant machine fleet support is planned for future iterations.
- **Decision Logic Simplification:** The system uses a union (OR) ensemble rule to maximize recall for safety. Future iterations can incorporate adaptive voting thresholds or cost-sensitive weighted scoring.
- **Hardware Integration:** Telemetry is generated via replay simulation of empirical physics data rather than live hardware sensor protocols (e.g. MQTT / Modbus / OPC-UA).
