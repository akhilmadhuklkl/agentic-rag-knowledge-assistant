from agentic_rag.api.app import create_app


def test_health_endpoint_reports_ready_application() -> None:
    app = create_app()
    health_route = next(route for route in app.routes if route.path == "/health")

    assert health_route.endpoint() == {"status": "ok", "environment": "development"}
