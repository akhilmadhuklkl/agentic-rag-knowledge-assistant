from pathlib import Path

from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteAuditRepository, SQLiteDocumentRepository


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

    repository.record("request-1", "GET", "/metrics", "Invalid API key")

    events = repository.recent()
    assert events[0]["request_id"] == "request-1"
    assert events[0]["reason"] == "Invalid API key"
    assert repository.recent(query="API key")[0]["path"] == "/metrics"
