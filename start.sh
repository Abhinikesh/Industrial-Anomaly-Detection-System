#!/usr/bin/env bash

# ==============================================================================
# Industrial Anomaly Detection System — All-in-One Startup Script
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "⚙️  Starting Industrial Anomaly Detection System"
echo "======================================================================"

# 1. Check MongoDB
echo "[1/4] Checking MongoDB service..."
if command -v mongosh &> /dev/null; then
  if mongosh --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
    echo "  ✓ MongoDB is running locally."
  else
    echo "  ⚠️  MongoDB is not responding. Starting MongoDB service..."
    if command -v brew &> /dev/null; then
      brew services start mongodb-community || true
    fi
  fi
else
  echo "  ℹ️  mongosh CLI not found. Assuming MongoDB is active at localhost:27017."
fi

# 2. Setup / Activate Backend
echo ""
echo "[2/4] Initializing Python Virtual Environment..."
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  echo "  Installing dependencies from requirements.txt..."
  pip install -r requirements.txt -q
else
  source venv/bin/activate
fi

# Check if model files exist; if not, prompt to train
if [ ! -f "$PROJECT_ROOT/models/isolation_forest.pkl" ] || [ ! -f "$PROJECT_ROOT/models/autoencoder.pt" ]; then
  echo "  ⚠️  Trained model files not found in models/ directory."
  if [ ! -f "$PROJECT_ROOT/data/raw/ai4i2020.csv" ]; then
    echo "  Downloading AI4I 2020 dataset from UCI repository..."
    python ml/download_dataset.py
  fi
  echo "  Training Isolation Forest baseline..."
  python ml/train_isolation_forest.py
  echo "  Training Deep Autoencoder baseline..."
  python ml/train_autoencoder.py
fi

# Start FastAPI Backend in Background
echo "  Starting FastAPI backend on http://localhost:8000 ..."
uvicorn app.main:app --port 8000 &
BACKEND_PID=$!
echo "  ✓ FastAPI backend running (PID: $BACKEND_PID)"

# 3. Setup / Start React Frontend
echo ""
echo "[3/4] Starting React Dashboard on http://localhost:3000 ..."
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
  echo "  Installing frontend dependencies (npm install)..."
  npm install --silent
fi

npm run dev -- --port 3000 &
FRONTEND_PID=$!
echo "  ✓ React frontend running (PID: $FRONTEND_PID)"

# 4. Prompt for Simulator
echo ""
echo "======================================================================"
echo "🚀 System is Live!"
echo "   - React Dashboard:  http://localhost:3000"
echo "   - FastAPI API Docs: http://localhost:8000/docs"
echo "======================================================================"
echo ""
echo "To start streaming telemetry in this terminal, press [ENTER]."
echo "Or open a separate terminal and run: cd backend && source venv/bin/activate && python ml/simulator.py"
echo ""

cleanup() {
  echo ""
  echo "Shutting down servers..."
  kill $BACKEND_PID 2>/dev/null || true
  kill $FRONTEND_PID 2>/dev/null || true
  echo "Done."
  exit 0
}

trap cleanup SIGINT SIGTERM

cd "$PROJECT_ROOT/backend"
read -r -p "Press [ENTER] to launch simulator now (or Ctrl-C to exit): " _
python ml/simulator.py
