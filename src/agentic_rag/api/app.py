"""FastAPI application entry point."""

import logging
import sqlite3
from pathlib import Path

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
    if settings.app_env.lower() != "development" and not settings.admin_user_ids:
        logging.getLogger("agentic_rag").warning(
            "No admin_user_ids configured; administrative audit views are unavailable"
        )
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
        if request.url.path not in {"/health", "/health/readiness"}:
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
                    get_request_id(),
                    request.method,
                    request.url.path,
                    str(error),
                    "high" if "API key" in str(error) else "medium",
                )
                audit_repository.purge_older_than(settings.audit_retention_days)
                status_code = 401 if "API key" in str(error) else 403
                return JSONResponse(status_code=status_code, content={"detail": str(error)})
            log_event(
                "request_authenticated",
                user_id=identity.user_id,
                workspace_id=identity.workspace_id,
            )
            request.state.user_id = identity.user_id
            request.state.workspace_id = identity.workspace_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        log_event("http_request_completed", method=request.method, path=request.url.path)
        return response

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": settings.app_env}

    @app.get("/health/readiness", tags=["system"])
    def readiness() -> dict[str, object]:
        """Report operational dependencies needed for local readiness."""

        components: dict[str, str] = {}
        try:
            with sqlite3.connect(settings.database_path) as connection:
                connection.execute("SELECT 1")
            components["database"] = "ready"
            components["audit_storage"] = "ready"
        except sqlite3.Error:
            components["database"] = "degraded"
            components["audit_storage"] = "degraded"
        vector_path = Path(settings.vector_store_path)
        components["vector_index"] = "ready" if vector_path.is_file() and vector_path.with_suffix(".meta.json").is_file() else "not_created"
        if settings.llm_provider.lower() == "ollama" or settings.embedding_provider.lower() == "ollama":
            try:
                response = httpx.get(f"{settings.llm_base_url.rstrip('/')}/api/tags", timeout=2.0)
                components["ollama"] = "ready" if response.is_success else "degraded"
            except httpx.HTTPError:
                components["ollama"] = "degraded"
        else:
            components["ollama"] = "not_required"
        status = "ready" if all(value in {"ready", "not_required"} for value in components.values()) else "degraded"
        return {"status": status, "environment": settings.app_env, "components": components}

    @app.get("/metrics", tags=["system"])
    def application_metrics() -> dict[str, object]:
        """Return local process metrics for development observability."""

        return metrics.snapshot()

    return app


app = create_app()
