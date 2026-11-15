from pathlib import Path

from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteDocumentRepository


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
