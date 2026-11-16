import pytest

from agentic_rag.config import Settings


def test_settings_have_safe_local_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.app_env == "development"
    assert settings.llm_provider == "deterministic"
    assert settings.embedding_provider == "local"
    assert settings.minimum_evidence_score == 0.4


def test_settings_validate_audit_retention() -> None:
    with pytest.raises(ValueError, match="audit_retention_days"):
        Settings(_env_file=None, audit_retention_days=0)


def test_settings_validate_admin_user_ids() -> None:
    settings = Settings(_env_file=None, admin_user_ids="admin_1,admin-2")
    assert settings.admin_user_ids == "admin_1,admin-2"
    with pytest.raises(ValueError, match="admin_user_ids"):
        Settings(_env_file=None, admin_user_ids="admin@example.com")
