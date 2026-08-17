from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Industrial Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: import and include routers once routes/ modules are ready

@app.get("/")
def root():
    return {"status": "Anomaly Detection API running"}

@app.get("/health")
def health():
    return {"status": "ok"}
