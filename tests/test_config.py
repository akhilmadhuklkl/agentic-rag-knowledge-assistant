from agentic_rag.config import Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings()
    assert settings.app_env == "development"
    assert settings.llm_provider == "deterministic"
    assert settings.embedding_provider == "local"
