"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the local MVP."""

    app_name: str = "Agentic RAG Knowledge Assistant"
    app_env: str = "development"
    log_level: str = "INFO"
    llm_provider: str = "local"
    embedding_provider: str = "local"
    vector_store_path: str = "data/vectorstore"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
