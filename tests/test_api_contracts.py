from fastapi import FastAPI

from agentic_rag.api.app import create_app


def test_api_registers_product_routes_and_error_handlers() -> None:
    app: FastAPI = create_app()
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/knowledge/ask" in paths
    assert "/knowledge/documents/upload" in paths
    assert ValueError in app.exception_handlers
    assert FileNotFoundError in app.exception_handlers
