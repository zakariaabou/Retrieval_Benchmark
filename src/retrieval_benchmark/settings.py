from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RB_", extra="ignore")
    database_url: str = "postgresql+asyncpg://benchmark:benchmark@localhost:5432/benchmark"
    mlflow_tracking_uri: str = "http://localhost:5000"
    default_chunking: str = "fixed_256_o32"
    default_strategy: str = "hybrid"
    cache_directory: Path = Path(".cache/retrieval-benchmark")
    data_directory: Path = Path("datasets")
    reports_directory: Path = Path("reports/generated")
    worker_poll_seconds: float = 1.0
    use_database: bool = False
    openai_api_key: str | None = None
    cohere_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
