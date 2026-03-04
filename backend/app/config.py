"""
Application configuration loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central config – override any value via .env or environment."""

    # Server
    app_title: str = "Smartphone Addiction Predictor"
    app_version: str = "1.0.0"
    debug: bool = False

    # Paths
    model_path: str = str(BASE_DIR / "models" / "addiction_model.pkl")
    metrics_path: str = str(BASE_DIR / "models" / "metrics.json")
    dataset_path: str = str(BASE_DIR / "data" / "smartphone_addiction.csv")

    # ML
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    ollama_timeout: int = 30

    # Upload
    max_upload_size_mb: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
