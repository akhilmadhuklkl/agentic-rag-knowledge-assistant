"""FastAPI application entry point."""

from fastapi import FastAPI, Request

from agentic_rag import __version__
from agentic_rag.api.routes import router as knowledge_router
from agentic_rag.config import get_settings
from agentic_rag.observability import (
    REQUEST_ID_HEADER,
    configure_logging,
    create_request_id,
    log_event,
    set_request_id,
)


def create_app() -> FastAPI:
    """Create the API application without relying on global UI state."""

    settings = get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(knowledge_router)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or create_request_id()
        set_request_id(request_id)
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        log_event("http_request_completed", method=request.method, path=request.url.path)
        return response

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    return app


app = create_app()
