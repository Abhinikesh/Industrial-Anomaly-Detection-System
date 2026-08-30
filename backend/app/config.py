"""
backend/app/config.py
=====================
Centralised application settings.

All values can be overridden via environment variables or a .env file
(loaded automatically by pydantic-settings).

Usage anywhere in the codebase:
    from app.config import settings

    uri = settings.mongo_uri
    port = settings.api_port
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # ── MongoDB ────────────────────────────────────────────────────────────────
    mongo_uri: str = Field(
        default="mongodb://localhost:27017/anomaly_detection",
        description="Full MongoDB connection URI including database name.",
    )
    mongo_server_selection_timeout_ms: int = Field(
        default=3000,
        description="Milliseconds before MongoClient raises ServerSelectionTimeoutError.",
    )

    # ── API ────────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="Uvicorn bind host.")
    api_port: int = Field(default=8000, description="Uvicorn bind port.")
    api_reload: bool = Field(default=False, description="Enable hot-reload (dev only).")
    cors_origins: str = Field(
        default="*",
        description="Comma-separated list of allowed CORS origins, or '*' for all.",
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Minimum log level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    log_file: str = Field(
        default="logs/app.log",
        description="Path to the rotating log file (relative to backend/).",
    )
    log_max_bytes: int = Field(
        default=10_485_760,  # 10 MB
        description="Maximum size of a single log file before rotation.",
    )
    log_backup_count: int = Field(
        default=5,
        description="Number of rotated log files to keep.",
    )

    # ── Model scoring ──────────────────────────────────────────────────────────
    # These are global safety guards; per-type thresholds come from the
    # saved autoencoder_threshold.json artefacts produced during training.
    iso_score_upper_bound: float = Field(
        default=1.0,
        description=(
            "Isolation Forest scores above this floor will be considered anomalous "
            "regardless of the trained decision boundary (emergency override)."
        ),
    )
    ae_score_upper_bound: float = Field(
        default=100.0,
        description="Reconstruction error values above this cap are clamped for logging.",
    )

    # ── Ingest validation ──────────────────────────────────────────────────────
    required_sensor_keys: dict = Field(
        default={
            "milling_machine": [
                "air_temp", "process_temp", "rpm", "torque", "tool_wear"
            ],
            "fleet_machine": [
                "voltage", "rotation", "pressure", "vibration"
            ],
            "water_pump": [
                "sensor_04", "sensor_00", "sensor_10", "sensor_06",
                "sensor_11", "sensor_07", "sensor_02", "sensor_08",
                "sensor_12", "sensor_09", "sensor_01", "sensor_03",
                "sensor_05", "sensor_40", "sensor_48",
            ],
        },
        description=(
            "Per-machine-type required sensor key lists for ingest validation. "
            "Incoming sensor_values dicts must contain at minimum these keys."
        ),
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",       # silently ignore unknown env vars
        case_sensitive=False, # MONGO_URI == mongo_uri
    )


# Singleton — imported everywhere
settings = Settings()
