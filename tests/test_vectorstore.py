from pathlib import Path

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.ingestion.service import DocumentIngestionService
from agentic_rag.vectorstore.store import LocalVectorStore

RUNTIME = Path(__file__).parent / ".runtime"


def test_vector_store_persists_and_returns_ranked_chunks() -> None:
    chunks = DocumentIngestionService().ingest_file(
        Path(__file__).parent / "fixtures" / "policy.md"
    )
    store_path = RUNTIME / "vectors-search.json"
    store_path.unlink(missing_ok=True)
    store = LocalVectorStore(store_path, LocalHashEmbeddingProvider())

    store.add(chunks)
    restored = LocalVectorStore(store_path, LocalHashEmbeddingProvider())
    results = restored.search("refund policy", top_k=1)

    assert restored.count() == 1
    assert len(results) == 1
    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].score > 0


def test_vector_store_upserts_same_chunk_id() -> None:
    chunks = DocumentIngestionService().ingest_file(
        Path(__file__).parent / "fixtures" / "policy.md"
    )
    store_path = RUNTIME / "vectors-upsert.json"
    store_path.unlink(missing_ok=True)
    store = LocalVectorStore(store_path, LocalHashEmbeddingProvider())

    store.add(chunks)
    store.add(chunks)

    assert store.count() == len(chunks)


def test_vector_store_rejects_a_different_embedding_provider(tmp_path: Path) -> None:
    store_path = tmp_path / "vectors.json"
    LocalVectorStore(store_path, LocalHashEmbeddingProvider()).add([])

    class DifferentProvider:
        provider_id = "different-provider"

        def embed(self, text: str) -> list[float]:
            return [1.0]

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [[1.0] for _ in texts]

    try:
        LocalVectorStore(store_path, DifferentProvider())
    except ValueError as error:
        assert "re-index" in str(error)
    else:
        raise AssertionError("Expected incompatible embedding provider to be rejected")
