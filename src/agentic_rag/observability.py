"""Request correlation and structured application logging."""

import json
import logging
from contextvars import ContextVar
from uuid import uuid4

REQUEST_ID_HEADER = "X-Request-ID"
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
logger = logging.getLogger("agentic_rag")


def create_request_id() -> str:
    """Create a short correlation identifier."""

    return uuid4().hex


def set_request_id(request_id: str) -> None:
    """Set the request ID for the current execution context."""

    _request_id.set(request_id)


def get_request_id() -> str | None:
    """Return the current request correlation identifier."""

    return _request_id.get()


def configure_logging(level: str = "INFO") -> None:
    """Configure a compact JSON logger for local and service logs."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")


def log_event(event: str, **fields: object) -> None:
    """Write one structured event with the active request ID."""

    payload = {"event": event, "request_id": get_request_id(), **fields}
    logger.info(json.dumps(payload, default=str))
