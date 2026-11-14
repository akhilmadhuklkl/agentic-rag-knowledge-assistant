from pathlib import Path

from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def test_indexing_service_connects_ingestion_and_vector_storage() -> None:
    store_path = RUNTIME / "indexing.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)

    summary = service.index_file(FIXTURES / "policy.md")

    assert summary.source_count == 1
    assert summary.chunk_count == 1
    assert service.vector_store.count() == 1
    assert service.vector_store.search("refund policy", top_k=1)[0].chunk.source_name == "policy.md"


def test_directory_indexing_is_repeatable() -> None:
    store_path = RUNTIME / "directory-indexing.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)

    first = service.index_directory(FIXTURES / "directory")
    second = service.index_directory(FIXTURES / "directory")

    assert first.source_count == 2
    assert first.chunk_count == 2
    assert second.chunk_count == 2
    assert service.vector_store.count() == 2
