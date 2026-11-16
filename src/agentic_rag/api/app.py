"""FastAPI application entry point."""

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag import __version__
from agentic_rag.api.routes import router as knowledge_router
from agentic_rag.config import get_settings
from agentic_rag.metrics import metrics
from agentic_rag.observability import (
    REQUEST_ID_HEADER,
    configure_logging,
    create_request_id,
    get_request_id,
    log_event,
    set_request_id,
)
from agentic_rag.security.auth import authenticate_request
from agentic_rag.storage.repository import SQLiteAuditRepository


def create_app() -> FastAPI:
    """Create the API application without relying on global UI state."""

    settings = get_settings()
    audit_repository = SQLiteAuditRepository(settings.database_path)
    configure_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, version=__version__)
    app.include_router(knowledge_router)
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="primary-ui")

    @app.get("/", include_in_schema=False)
    def primary_ui() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, error: ValueError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(error)})

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, error: FileNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.exception_handler(httpx.HTTPError)
    async def upstream_error_handler(request: Request, error: httpx.HTTPError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": "Upstream model service failed"})

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or create_request_id()
        set_request_id(request_id)
        if request.url.path != "/health":
            try:
                identity = authenticate_request(dict(request.headers), settings)
            except PermissionError as error:
                log_event(
                    "request_authentication_failed",
                    method=request.method,
                    path=request.url.path,
                    reason=str(error),
                )
                audit_repository.record(
                    get_request_id(), request.method, request.url.path, str(error)
                )
                status_code = 401 if "API key" in str(error) else 403
                return JSONResponse(status_code=status_code, content={"detail": str(error)})
            log_event(
                "request_authenticated",
                user_id=identity.user_id,
                workspace_id=identity.workspace_id,
            )
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        log_event("http_request_completed", method=request.method, path=request.url.path)
        return response

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    @app.get("/metrics", tags=["system"])
    def application_metrics() -> dict[str, object]:
        """Return local process metrics for development observability."""

        return metrics.snapshot()

    return app


app = create_app()
