"""FastAPI application entry point."""

import asyncio
import csv
import io
import json
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from agentic_rag import __version__
from agentic_rag.api.routes import _require_admin, ingestion_repository
from agentic_rag.api.routes import router as knowledge_router
from agentic_rag.config import Settings, get_settings
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
from agentic_rag.security.rate_limit import InMemoryRateLimiter
from agentic_rag.security.request_context import (
    apply_security_headers,
    classify_authentication_failure,
    is_public_health_path,
)
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteAuditRepository, SQLiteOperationalRepository


async def recycle_bin_cleanup_loop(settings: Settings, status: dict[str, object], operational_repository: SQLiteOperationalRepository | None = None) -> None:
    """Periodically purge expired recycle-bin files until application shutdown."""

    while True:
        await asyncio.sleep(settings.recycle_bin_cleanup_interval_seconds)
        recycle_bin_cleanup_once(settings, status, operational_repository)


def recycle_bin_cleanup_once(settings: Settings, status: dict[str, object], operational_repository: SQLiteOperationalRepository | None = None) -> list[str]:
    """Run one cleanup cycle and update the observable worker status."""

    status["last_started_at"] = time.time()
    started = time.perf_counter()
    try:
        storage = DocumentStorageService(settings.document_storage_path)
        removed = storage.purge_recycle_bin(settings.recycle_bin_retention_days)
        for name in removed:
            ingestion_repository.record("purged", name, f"Automatic cleanup after {settings.recycle_bin_retention_days} days")
        status["last_run_at"] = time.time()
        status["last_removed_count"] = len(removed)
        status["last_error"] = None
        metrics.increment("recycle_bin_cleanup_runs_total")
        metrics.increment("recycle_bin_items_removed_total", len(removed))
        duration_ms = (time.perf_counter() - started) * 1000
        if operational_repository:
            operational_repository.record_cleanup("success", len(removed), removed_names=removed, duration_ms=duration_ms)
        if removed:
            logging.getLogger("agentic_rag").info("recycle_bin_cleanup_completed", extra={"removed": len(removed)})
        return removed
    except Exception as error:  # pragma: no cover - defensive worker boundary
        status["last_error"] = str(error)
        metrics.increment("recycle_bin_cleanup_failures_total")
        duration_ms = (time.perf_counter() - started) * 1000
        if operational_repository:
            operational_repository.record_cleanup("error", 0, str(error), [], duration_ms)
        logging.getLogger("agentic_rag").exception("recycle_bin_cleanup_failed")
        return []


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API application without relying on global UI state."""

    settings = settings or get_settings()
    if settings.app_env.lower() != "development" and not settings.admin_user_ids:
        logging.getLogger("agentic_rag").warning(
            "No admin_user_ids configured; administrative audit views are unavailable"
        )
    audit_repository = SQLiteAuditRepository(settings.database_path)
    operational_repository = SQLiteOperationalRepository(settings.database_path)
    rate_limiter = InMemoryRateLimiter()
    started_at = time.monotonic()
    cleanup_status: dict[str, object] = {"last_started_at": None, "last_run_at": None, "last_removed_count": 0, "last_error": None}
    configure_logging(settings.log_level)
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        cleanup_task = asyncio.create_task(recycle_bin_cleanup_loop(settings, cleanup_status, operational_repository))
        try:
            yield
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    app.include_router(knowledge_router)
    app.mount("/app", StaticFiles(directory="frontend", html=True), name="primary-ui")

    @app.get("/health/recycle-bin", tags=["system"])
    def recycle_bin_health() -> dict[str, object]:
        """Report automatic recycle-bin cleanup state without file contents."""

        return {
            "status": "error" if cleanup_status["last_error"] else "scheduled",
            "retention_days": settings.recycle_bin_retention_days,
            "interval_seconds": settings.recycle_bin_cleanup_interval_seconds,
            "last_started_at": cleanup_status["last_started_at"],
            "last_run_at": cleanup_status["last_run_at"],
            "last_removed_count": cleanup_status["last_removed_count"],
            "last_error": cleanup_status["last_error"],
            "history": operational_repository.recent_cleanup(10),
        }

    @app.post("/health/recycle-bin/run", tags=["system"])
    def run_recycle_bin_cleanup(request: Request) -> dict[str, object]:
        """Run recycle-bin cleanup immediately for an authorized administrator."""

        admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
        _require_admin(request, admins)
        removed = recycle_bin_cleanup_once(settings, cleanup_status, operational_repository)
        return {
            "status": "error" if cleanup_status["last_error"] else "completed",
            "removed": removed,
            "removed_count": len(removed),
            "error": cleanup_status["last_error"],
        }

    @app.get("/health/recycle-bin/export", tags=["system"])
    def export_recycle_bin_cleanup(request: Request, format: str = Query(default="json", pattern="^(json|csv)$"), from_date: str | None = Query(default=None), to_date: str | None = Query(default=None), status: str = Query(default="all", pattern="^(all|success|error)$")) -> PlainTextResponse:
        """Export persisted recycle-bin cleanup history for administrators."""

        admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
        _require_admin(request, admins)
        history = operational_repository.recent_cleanup(1000)
        if from_date or to_date or status != "all":
            start = datetime.fromisoformat(from_date).date() if from_date else None
            end = datetime.fromisoformat(to_date).date() if to_date else None
            history = [item for item in history if (status == "all" or item["status"] == status) and (not start or datetime.fromisoformat(f"{item['created_at']}+00:00").date() >= start) and (not end or datetime.fromisoformat(f"{item['created_at']}+00:00").date() <= end)]
        if format == "json":
            content = json.dumps(history, indent=2)
            media_type = "application/json"
        else:
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["status", "removed_count", "removed_names", "error", "created_at"])
            for item in history:
                writer.writerow([item["status"], item["removed_count"], "; ".join(item["removed_names"]), item["error"] or "", item["created_at"]])
            content = output.getvalue()
            media_type = "text/csv"
        return PlainTextResponse(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="atlas-cleanup-history.{format}"'})

    @app.get("/health/recycle-bin/history", tags=["system"])
    def recycle_bin_cleanup_history(request: Request, limit: int = Query(default=10, ge=1, le=100), offset: int = Query(default=0, ge=0)) -> dict[str, object]:
        """Return a bounded page of cleanup history for report navigation."""

        admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
        _require_admin(request, admins)
        page = operational_repository.recent_cleanup(limit + 1)[offset:offset + limit + 1]
        return {"items": page[:limit], "has_more": len(page) > limit, "offset": offset, "limit": limit}

    @app.get("/health/recycle-bin/summary", tags=["system"])
    def recycle_bin_cleanup_summary(request: Request) -> dict[str, object]:
        """Return aggregate cleanup run and reliability statistics."""

        admins = {value.strip() for value in settings.admin_user_ids.split(",") if value.strip()}
        _require_admin(request, admins)
        return operational_repository.cleanup_summary()

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
        if not is_public_health_path(request.url.path):
            client_key = request.client.host if request.client else "unknown"
            limit = settings.upload_rate_limit_per_minute if request.url.path.endswith("/documents/upload") else settings.rate_limit_per_minute
            if not rate_limiter.allow(client_key, limit):
                metrics.increment("http_requests_total")
                metrics.increment("http_errors_total")
                metrics.increment("rate_limit_rejections_total")
                response = JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"}, headers={"Retry-After": "60"})
                response.headers[REQUEST_ID_HEADER] = request_id
                apply_security_headers(response)
                return response
            try:
                identity = authenticate_request(dict(request.headers), settings)
            except PermissionError as error:
                failure = classify_authentication_failure(error)
                metrics.increment("http_requests_total")
                metrics.increment("http_errors_total")
                metrics.increment("authentication_failures_total")
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
                    failure.severity,
                )
                audit_repository.purge_older_than(settings.audit_retention_days)
                return JSONResponse(status_code=failure.status_code, content={"detail": failure.reason})
            log_event(
                "request_authenticated",
                user_id=identity.user_id,
                workspace_id=identity.workspace_id,
            )
            request.state.user_id = identity.user_id
            request.state.workspace_id = identity.workspace_id
        response = await call_next(request)
        metrics.increment("http_requests_total")
        if response.status_code >= 500:
            metrics.increment("http_errors_total")
        response.headers[REQUEST_ID_HEADER] = request_id
        apply_security_headers(response)
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
        duration = metrics.snapshot().get("durations", {}).get("assistant_workflow", {})
        counters = metrics.snapshot().get("counters", {})
        request_count = int(counters.get("http_requests_total", 0))
        error_count = int(counters.get("http_errors_total", 0))
        error_rate = round(error_count / request_count, 4) if request_count else 0.0
        alert = error_rate >= settings.error_rate_alert_threshold and request_count > 0
        if alert:
            status = "degraded"
        try:
            operational_repository.record(status, time.monotonic() - started_at, error_rate)
            operational_repository.purge_older_than(settings.audit_retention_days)
        except sqlite3.Error:
            logging.getLogger("agentic_rag").warning("Operational snapshot could not be persisted")
        return {
            "status": status,
            "environment": settings.app_env,
            "components": components,
            "uptime_seconds": round(time.monotonic() - started_at, 1),
            "latency": {
                "sample_count": duration.get("count", 0),
                "average_ms": duration.get("average_ms"),
            },
            "error_rate": error_rate,
            "error_count": error_count,
            "request_count": request_count,
            "alert": "Elevated error rate" if alert else None,
        }

    @app.get("/health/history", tags=["system"])
    def health_history(limit: int = 20) -> list[dict[str, object]]:
        """Return recent persisted readiness snapshots."""

        return operational_repository.recent(max(1, min(limit, 100)))

    @app.get("/health/diagnostics", tags=["system"])
    def diagnostics() -> dict[str, object]:
        """Return a safe, export-friendly local diagnostics snapshot."""

        return {
            "readiness": readiness(),
            "history": operational_repository.recent(20),
            "metrics": metrics.snapshot(),
        }

    @app.get("/metrics", tags=["system"])
    def application_metrics() -> dict[str, object]:
        """Return local process metrics for development observability."""

        return metrics.snapshot()

    return app


app = create_app()
