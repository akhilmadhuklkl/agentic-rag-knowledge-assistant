"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import field_validator
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
    web_search_enabled: bool = False
    web_search_timeout_seconds: float = 4.0
    web_search_allowed_domains: str = ""
    audit_retention_days: int = 30
    recycle_bin_retention_days: int = 30
    recycle_bin_cleanup_interval_seconds: int = 86_400
    recycle_bin_cleanup_alert_threshold_ms: int = 5_000
    admin_user_ids: str = ""
    error_rate_alert_threshold: float = 0.1
    rate_limit_per_minute: int = 120
    upload_rate_limit_per_minute: int = 10

    @field_validator("audit_retention_days")
    @classmethod
    def validate_audit_retention_days(cls, value: int) -> int:
        if not 1 <= value <= 3650:
            raise ValueError("audit_retention_days must be between 1 and 3650")
        return value

    @field_validator("recycle_bin_retention_days")
    @classmethod
    def validate_recycle_bin_retention_days(cls, value: int) -> int:
        if not 1 <= value <= 3650:
            raise ValueError("recycle_bin_retention_days must be between 1 and 3650")
        return value

    @field_validator("recycle_bin_cleanup_interval_seconds")
    @classmethod
    def validate_recycle_bin_cleanup_interval_seconds(cls, value: int) -> int:
        if not 60 <= value <= 604_800:
            raise ValueError("recycle_bin_cleanup_interval_seconds must be between 60 and 604800")
        return value

    @field_validator("recycle_bin_cleanup_alert_threshold_ms")
    @classmethod
    def validate_recycle_bin_cleanup_alert_threshold_ms(cls, value: int) -> int:
        if not 100 <= value <= 3_600_000:
            raise ValueError("recycle_bin_cleanup_alert_threshold_ms must be between 100 and 3600000")
        return value

    @field_validator("admin_user_ids")
    @classmethod
    def validate_admin_user_ids(cls, value: str) -> str:
        identifiers = [item.strip() for item in value.split(",") if item.strip()]
        for identifier in identifiers:
            if not identifier.replace("_", "a").replace("-", "a").isalnum():
                raise ValueError("admin_user_ids must contain only letters, numbers, '_' or '-'")
        return ",".join(identifiers)

    @field_validator("error_rate_alert_threshold")
    @classmethod
    def validate_error_rate_alert_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("error_rate_alert_threshold must be between 0 and 1")
        return value

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
