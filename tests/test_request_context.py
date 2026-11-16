import pytest

from agentic_rag.security.request_context import (
    classify_authentication_failure,
    is_public_health_path,
)


def test_public_health_path_rules_are_explicit() -> None:
    assert is_public_health_path("/health")
    assert is_public_health_path("/health/readiness")
    assert not is_public_health_path("/health/history")
    assert not is_public_health_path("/health/diagnostics")


@pytest.mark.parametrize(
    ("message", "status_code", "severity"),
    [("Invalid API key", 401, "high"), ("Workspace access denied", 403, "medium")],
)
def test_authentication_failure_classification(message: str, status_code: int, severity: str) -> None:
    failure = classify_authentication_failure(PermissionError(message))

    assert failure.status_code == status_code
    assert failure.severity == severity
    assert failure.reason == message
