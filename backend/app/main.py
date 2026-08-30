from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.logger import get_logger
from app.routes.ingest import router as ingest_router
from app.routes.readings import router as readings_router

log = get_logger(__name__)

app = FastAPI(
    title="Industrial Anomaly Detection API",
    description=(
        "Real-time predictive maintenance API supporting three machine types: "
        "milling_machine (AI4I 2020), fleet_machine (Azure PdM), and water_pump "
        "(Pump Sensor Data). Scores incoming telemetry with Isolation Forest + "
        "PyTorch Autoencoder and stores results in MongoDB."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(readings_router)

# Mount results directory to serve static evaluation plots
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../../results")
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.on_event("startup")
async def on_startup():
    log.info(
        "Anomaly Detection API v2.0 starting — host=%s port=%d",
        settings.api_host,
        settings.api_port,
    )


@app.on_event("shutdown")
async def on_shutdown():
    log.info("Anomaly Detection API shutting down.")


@app.get("/")
def root():
    return {"status": "Anomaly Detection API running", "version": "2.0.0"}


@app.get("/health")
def health():
    return {"status": "ok"}
