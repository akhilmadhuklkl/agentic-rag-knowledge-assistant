"""FastAPI application entry point."""

from fastapi import FastAPI

from agentic_rag import __version__
from agentic_rag.api.routes import router as knowledge_router
from agentic_rag.config import get_settings


def create_app() -> FastAPI:
    """Create the API application without relying on global UI state."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(knowledge_router)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return app


app = create_app()
