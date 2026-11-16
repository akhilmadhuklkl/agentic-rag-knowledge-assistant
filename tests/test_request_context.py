import pytest

from agentic_rag.security.request_context import (
    apply_security_headers,
    classify_authentication_failure,
    is_public_health_path,
)


class ResponseStub:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


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


def test_security_headers_are_applied() -> None:
    response = ResponseStub()

    apply_security_headers(response)

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
