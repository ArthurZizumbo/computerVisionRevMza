"""
Configuration management using Pydantic Settings.

This module provides centralized configuration for the application
with environment variable support and validation.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Attributes
    ----------
    google_maps_api_key : str
        Google Maps Static API key
    gcp_project_id : str
        Google Cloud project ID
    mlflow_tracking_uri : str
        MLflow tracking server URI
    log_level : str
        Logging level
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    google_maps_api_key: str = Field(
        default="",
        description="Google Maps Static API key",
    )

    gcp_project_id: str = Field(
        default="geo-rect-prod",
        description="GCP project ID",
    )

    gcp_region: str = Field(
        default="us-central1",
        description="GCP region",
    )

    gcs_bucket: str = Field(
        default="geo-rect-artifacts",
        description="GCS bucket for artifacts",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking URI",
    )

    dvc_remote: str = Field(
        default="gs://geo-rect-artifacts",
        description="DVC remote storage URI",
    )

    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    data_dir: Path = Field(
        default=Path("data"),
        description="Data directory path",
    )

    models_dir: Path = Field(
        default=Path("models"),
        description="Models directory path",
    )

    cache_dir: Path = Field(
        default=Path("data/cache"),
        description="Cache directory path",
    )

    device: str | None = Field(
        default=None,
        description="Device for inference (cuda/cpu)",
    )

    ecc_threshold: float = Field(
        default=0.85,
        description="ECC confidence threshold",
    )

    loftr_min_matches: int = Field(
        default=50,
        description="Minimum LoFTR matches",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns
    -------
    Settings
        Application settings instance
    """
    return Settings()
