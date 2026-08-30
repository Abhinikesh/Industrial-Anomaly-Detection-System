#!/usr/bin/env bash

# ==============================================================================
# Industrial Anomaly Detection System — All-in-One Startup Script
# ==============================================================================
# Starts MongoDB (if not running), FastAPI backend, React frontend, and
# all three sensor stream simulators in background processes.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "⚙️  Industrial Anomaly Detection System v2.0"
echo "======================================================================"

# ── 1. Check MongoDB ──────────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking MongoDB service..."
if command -v mongosh &> /dev/null; then
  if mongosh --eval "db.adminCommand('ping')" --quiet &> /dev/null; then
    echo "  ✓ MongoDB is running."
  else
    echo "  ⚠️  MongoDB not responding — attempting to start..."
    if command -v brew &> /dev/null; then
      brew services start mongodb-community || true
    fi
  fi
else
  echo "  ℹ️  mongosh not found. Assuming MongoDB is active at localhost:27017."
fi

# ── 2. Python venv + deps ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Initialising Python virtual environment..."
cd "$PROJECT_ROOT/backend"
if [ ! -d "venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv venv
  source venv/bin/activate
  echo "  Installing dependencies..."
  pip install -r requirements.txt -q
else
  source venv/bin/activate
fi

# Ensure log directory exists before uvicorn starts
mkdir -p logs

# ── 3. Train models if missing ────────────────────────────────────────────────
echo ""
echo "[3/5] Checking model artefacts..."

train_if_missing() {
  local label="$1"; local marker="$2"; local download="$3"; local train1="$4"; local train2="$5"
  if [ ! -f "$PROJECT_ROOT/$marker" ]; then
    echo "  [$label] models not found — training now..."
    [ -n "$download" ] && python $download
    python $train1
    [ -n "$train2" ] && python $train2
    echo "  [$label] ✓ Training complete."
  else
    echo "  [$label] ✓ Models found."
  fi
}

train_if_missing "Milling Machine (AI4I)" \
  "models/isolation_forest.pkl" \
  "ml/download_dataset.py" \
  "ml/train_isolation_forest.py" \
  "ml/train_autoencoder.py"

train_if_missing "Azure PdM Fleet" \
  "models/azure/isolation_forest.pkl" \
  "" \
  "ml/train_azure_models.py" \
  ""

train_if_missing "Water Pump" \
  "models/pump/isolation_forest.pkl" \
  "" \
  "ml/train_pump_models.py" \
  ""

# ── 4. Start FastAPI backend ──────────────────────────────────────────────────
echo ""
echo "[4/5] Starting FastAPI backend (http://localhost:8000) ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
sleep 2
echo "  ✓ Backend PID $BACKEND_PID"

# ── 5. Start frontend ─────────────────────────────────────────────────────────
echo ""
echo "[5/5] Starting React frontend (http://localhost:3000) ..."
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
  echo "  Installing npm dependencies..."
  npm install -q
fi
npm run dev -- --port 3000 &
FRONTEND_PID=$!
sleep 2
echo "  ✓ Frontend PID $FRONTEND_PID"

# ── 6. Start all 3 simulators ─────────────────────────────────────────────────
echo ""
echo "Starting simulators..."
cd "$PROJECT_ROOT/backend"

python ml/simulator.py       &
MILLING_PID=$!
echo "  ⚙️  Milling Machine simulator PID $MILLING_PID"

python ml/simulator_azure.py &
AZURE_PID=$!
echo "  ☁️  Azure Fleet simulator    PID $AZURE_PID"

python ml/simulator_pump.py  &
PUMP_PID=$!
echo "  💧 Water Pump simulator     PID $PUMP_PID"

echo ""
echo "======================================================================"
echo "✅ System is running!"
echo ""
echo "  Dashboard : http://localhost:3000"
echo "  API Docs  : http://localhost:8000/docs"
echo "  Log file  : backend/logs/app.log"
echo ""
echo "Press Ctrl+C to stop all processes."
echo "======================================================================"

# Wait for any process to exit, then clean up all
trap "kill $BACKEND_PID $FRONTEND_PID $MILLING_PID $AZURE_PID $PUMP_PID 2>/dev/null; echo 'All processes stopped.'" EXIT
wait
