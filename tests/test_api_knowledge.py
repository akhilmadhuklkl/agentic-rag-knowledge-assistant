from pathlib import Path

from agentic_rag.api.routes import index_file, search
from agentic_rag.api.schemas import IndexPathRequest, SearchRequest
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def test_index_file_route_returns_index_summary() -> None:
    store_path = RUNTIME / "api-indexing.json"
    store_path.unlink(missing_ok=True)
    store_path.with_suffix(".manifest.json").unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)

    response = index_file(IndexPathRequest(path=str(FIXTURES / "policy.md")), service)

    assert response.source_count == 1
    assert response.chunk_count == 1


def test_search_route_returns_citation_ready_results() -> None:
    store_path = RUNTIME / "api-search.json"
    store_path.unlink(missing_ok=True)
    store_path.with_suffix(".manifest.json").unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_file(FIXTURES / "policy.md")

    response = search(SearchRequest(query="refund policy", top_k=1), service)

    assert len(response.results) == 1
    assert response.results[0].source_name == "policy.md"
    assert response.results[0].page_number is None
    assert response.results[0].score > 0


def test_search_route_accepts_source_filters() -> None:
    store_path = RUNTIME / "api-filter.json"
    store_path.unlink(missing_ok=True)
    store_path.with_suffix(".manifest.json").unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_directory(FIXTURES / "directory")

    response = search(
        SearchRequest(query="first", top_k=5, source_name="b.txt"),
        service,
    )

    assert response.results == []
