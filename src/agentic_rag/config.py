"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the local MVP."""

    app_name: str = "Agentic RAG Knowledge Assistant"
    app_env: str = "development"
    log_level: str = "INFO"
    llm_provider: str = "deterministic"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"
    embedding_provider: str = "local"
    embedding_model: str = "nomic-embed-text"
    vector_store_path: str = "data/vectorstore"
    document_storage_path: str = "data/documents"
    conversation_store_path: str = "data/conversations.json"
    database_path: str = "data/assistant.db"
    retrieval_semantic_weight: float = 0.7
    retrieval_rerank_weight: float = 0.08
    minimum_lexical_overlap: float = 0.3
    minimum_evidence_score: float = 0.4
    confidence_warning_threshold: float = 0.4
    response_latency_warning_ms: float = 10_000
    evaluation_regression_threshold: float = 0.8
    evaluation_retrieval_threshold: float = 0.6
    evaluation_citation_threshold: float = 0.6
    default_top_k: int = 5
    max_top_k: int = 50
    api_key: str | None = None
    workspace_id: str = "default"
    require_workspace_header: bool = False
    max_upload_size_bytes: int = 10_000_000
    max_question_length: int = 2_000
    llm_max_retries: int = 2

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
