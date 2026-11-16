"""Pure request-context rules used by the HTTP middleware."""

from dataclasses import dataclass

PUBLIC_HEALTH_PATHS = frozenset({"/health", "/health/readiness"})


@dataclass(frozen=True)
class AuthenticationFailure:
    """Safe classification for an authentication failure."""

    status_code: int
    severity: str
    reason: str


def is_public_health_path(path: str) -> bool:
    """Return whether a path may be accessed without request authentication."""

    return path in PUBLIC_HEALTH_PATHS


def classify_authentication_failure(error: PermissionError) -> AuthenticationFailure:
    """Map an auth error to an HTTP status and safe audit severity."""

    reason = str(error)
    invalid_key = "API key" in reason
    return AuthenticationFailure(
        status_code=401 if invalid_key else 403,
        severity="high" if invalid_key else "medium",
        reason=reason,
    )


def apply_security_headers(response: object) -> None:
    """Apply browser security headers to a response-like object."""

    headers = response.headers
    headers["X-Content-Type-Options"] = "nosniff"
    headers["X-Frame-Options"] = "DENY"
    headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
