from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routes.ingest import router as ingest_router
from app.routes.readings import router as readings_router

app = FastAPI(title="Industrial Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router)
app.include_router(readings_router)

# Mount results directory to serve static evaluation plots
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "../../results")
if os.path.exists(RESULTS_DIR):
    app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/")
def root():
    return {"status": "Anomaly Detection API running"}


@app.get("/health")
def health():
    return {"status": "ok"}
