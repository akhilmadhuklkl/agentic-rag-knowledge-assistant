from agentic_rag.api.app import create_app
from agentic_rag.api.routes import provider_health
from agentic_rag.config import Settings


def test_health_endpoint_reports_ready_application() -> None:
    app = create_app()
    health_route = next(route for route in app.routes if route.path == "/health")

    assert health_route.endpoint() == {"status": "ok", "environment": "development"}


def test_provider_health_reports_local_index_compatibility(tmp_path) -> None:
    health = provider_health(Settings(vector_store_path=str(tmp_path / "vectors")))

    assert health.status == "ready"
    assert health.index_status == "not_created"
    assert health.embedding_provider_id.startswith("local-hash-v1:")


def test_readiness_endpoint_is_registered() -> None:
    app = create_app()

    assert any(route.path == "/health/readiness" for route in app.routes)
