import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_rag.api.routes import security_audit
from agentic_rag.config import Settings
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import (
    SQLiteAuditRepository,
    SQLiteDocumentRepository,
    SQLiteOperationalRepository,
)


def test_sqlite_document_repository_stores_metadata_and_fingerprint(tmp_path: Path) -> None:
    database = tmp_path / "documents.db"
    repository = SQLiteDocumentRepository(database)
    document = DocumentStorageService(
        tmp_path / "repository-documents",
        repository=repository,
    ).save("policy.md", b"refund policy")

    repository.set_fingerprint(document.path, "hash-1")

    assert repository.list_documents()[0].name == "policy.md"
    assert repository.get_fingerprint(document.path) == "hash-1"


def test_sqlite_audit_repository_persists_recent_security_events(tmp_path: Path) -> None:
    repository = SQLiteAuditRepository(tmp_path / "assistant.db")

    repository.record("request-1", "GET", "/metrics", "Invalid API key", "high")
    repository.record("request-2", "GET", "/knowledge", "Workspace access denied", "medium")

    events = repository.recent()
    assert events[0]["request_id"] == "request-2"
    assert events[0]["reason"] == "Workspace access denied"
    assert events[0]["severity"] == "medium"
    assert repository.recent(query="API key")[0]["path"] == "/metrics"
    assert repository.recent(limit=1, offset=1)[0]["request_id"] == "request-1"


def test_sqlite_operational_repository_persists_readiness_snapshots(tmp_path: Path) -> None:
    repository = SQLiteOperationalRepository(tmp_path / "assistant.db")

    repository.record("ready", 12.5, 0.02)

    snapshot = repository.recent(limit=1)[0]
    assert snapshot["status"] == "ready"
    assert snapshot["error_rate"] == 0.02

    with sqlite3.connect(tmp_path / "assistant.db") as connection:
        connection.execute(
            "UPDATE operational_snapshots SET created_at = '2000-01-01 00:00:00'"
        )
    assert repository.purge_older_than(1) == 1
    assert repository.recent() == []


def test_security_audit_route_requires_an_admin_user() -> None:
    request = SimpleNamespace(state=SimpleNamespace(user_id="viewer"))

    with pytest.raises(PermissionError, match="Administrator access required"):
        security_audit(request, Settings(admin_user_ids="admin"))


def test_security_audit_route_allows_an_admin_user() -> None:
    request = SimpleNamespace(state=SimpleNamespace(user_id="admin"))

    result = security_audit(request, Settings(admin_user_ids="admin"), limit=1, offset=0, query=None)

    assert len(result) <= 1
