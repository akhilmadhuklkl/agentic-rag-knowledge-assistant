import json
import logging

from agentic_rag.observability import log_event, set_request_id


def test_authentication_audit_event_contains_request_context_without_secrets(caplog) -> None:
    set_request_id("request-123")

    with caplog.at_level(logging.INFO, logger="agentic_rag"):
        log_event(
            "request_authentication_failed",
            method="GET",
            path="/metrics",
            reason="Invalid API key",
        )

    event = json.loads(caplog.records[-1].message)
    assert event == {
        "event": "request_authentication_failed",
        "request_id": "request-123",
        "method": "GET",
        "path": "/metrics",
        "reason": "Invalid API key",
    }
    assert "secret" not in caplog.records[-1].message
