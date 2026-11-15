from pathlib import Path

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.ingestion.models import DocumentChunk
from agentic_rag.ingestion.service import DocumentIngestionService
from agentic_rag.retrieval.service import HybridRetrievalService
from agentic_rag.vectorstore.store import LocalVectorStore

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def build_retrieval_service() -> HybridRetrievalService:
    store_path = RUNTIME / "retrieval.json"
    store_path.unlink(missing_ok=True)
    store = LocalVectorStore(store_path, LocalHashEmbeddingProvider())
    chunks = DocumentIngestionService().ingest_directory(FIXTURES / "directory")
    store.add(chunks)
    return HybridRetrievalService(store)


def test_hybrid_retrieval_returns_ranked_evidence() -> None:
    results = build_retrieval_service().search("first", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.source_name == "a.md"
    assert results[0].lexical_score > 0
    assert results[0].score > 0


def test_hybrid_retrieval_rejects_invalid_queries() -> None:
    service = build_retrieval_service()

    try:
        service.search("   ")
    except ValueError as error:
        assert "query" in str(error)
    else:
        raise AssertionError("Expected empty query to be rejected")


def test_hybrid_retrieval_filters_by_source_metadata() -> None:
    results = build_retrieval_service().search("first", source_name="b.txt")

    assert results == []


def test_hybrid_retrieval_normalizes_file_type_filter() -> None:
    results = build_retrieval_service().search("first", file_type=".md")

    assert len(results) == 1
    assert results[0].chunk.file_type == "md"


def test_hybrid_retrieval_boosts_exact_multi_word_phrases() -> None:
    service = build_retrieval_service()
    exact = DocumentChunk("exact", "first second", "a.md", "a.md", "md", None, 0)
    separated = DocumentChunk("separated", "first item, then second", "b.txt", "b.txt", "txt", None, 0)

    assert service._phrase_score("first second", exact) == 1.0
    assert service._phrase_score("first second", separated) == 0.0


def test_hybrid_retrieval_reranks_close_query_terms() -> None:
    service = build_retrieval_service()
    close = DocumentChunk("close", "security training first week", "a.md", "a.md", "md", None, 0)
    distant = DocumentChunk("distant", "security overview with unrelated details and training", "b.txt", "b.txt", "txt", None, 0)

    assert service._term_proximity_score("security training", close) > service._term_proximity_score("security training", distant)
