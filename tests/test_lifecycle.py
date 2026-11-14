from pathlib import Path

from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.storage.documents import DocumentStorageService
from agentic_rag.storage.repository import SQLiteDocumentRepository

RUNTIME = Path(__file__).parent / ".runtime"


def test_delete_document_removes_local_file_and_metadata() -> None:
    root = RUNTIME / "lifecycle-documents"
    database = RUNTIME / "lifecycle.db"
    database.unlink(missing_ok=True)
    repository = SQLiteDocumentRepository(database)
    storage = DocumentStorageService(root, repository=repository)
    saved = storage.save("remove-me.md", b"temporary")

    deleted = storage.delete(saved.name)

    assert deleted.name == saved.name
    assert not Path(saved.path).exists()
    assert storage.list_documents() == []


def test_cleanup_removes_vectors_for_missing_sources() -> None:
    store_path = RUNTIME / "cleanup.json"
    store_path.unlink(missing_ok=True)
    store_path.with_suffix(".manifest.json").unlink(missing_ok=True)
    source = RUNTIME / "cleanup-source.md"
    source.write_text("cleanup source", encoding="utf-8")
    service = DocumentIndexingService(store_path)
    service.index_file(source)
    source.unlink()

    assert service.cleanup_missing_sources(RUNTIME) == 1
    assert service.vector_store.count() == 0
