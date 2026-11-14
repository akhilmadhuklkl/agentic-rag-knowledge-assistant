from pathlib import Path

from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteDocumentRepository


def test_sqlite_document_repository_stores_metadata_and_fingerprint() -> None:
    database = Path(__file__).parent / ".runtime" / "documents.db"
    database.unlink(missing_ok=True)
    repository = SQLiteDocumentRepository(database)
    document = DocumentStorageService(
        Path(__file__).parent / ".runtime" / "repository-documents",
        repository=repository,
    ).save("policy.md", b"refund policy")

    repository.set_fingerprint(document.path, "hash-1")

    assert repository.list_documents()[0].name == "policy.md"
    assert repository.get_fingerprint(document.path) == "hash-1"
