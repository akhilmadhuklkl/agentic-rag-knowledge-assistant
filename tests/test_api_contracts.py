from fastapi import FastAPI

from agentic_rag.api.app import create_app
from agentic_rag.security.request_context import PUBLIC_HEALTH_PATHS


def test_api_registers_product_routes_and_error_handlers() -> None:
    app: FastAPI = create_app()
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/metrics" in paths
    assert "/knowledge/ask" in paths
    assert "/knowledge/documents/upload" in paths
    assert "/health/recycle-bin" in paths
    assert "/health/recycle-bin/run" in paths
    assert "/health/recycle-bin/export" in paths
    assert "/health/recycle-bin/history" in paths
    assert ValueError in app.exception_handlers
    assert FileNotFoundError in app.exception_handlers


def test_api_security_contract_lists_diagnostics_as_protected() -> None:
    paths = {route.path for route in create_app().routes}

    assert {"/health", "/health/readiness"}.issubset(paths)
    assert {"/health/history", "/health/diagnostics"}.issubset(paths)
    assert PUBLIC_HEALTH_PATHS.isdisjoint({"/health/history", "/health/diagnostics"})
