"""Application service connecting document ingestion to vector storage."""

from dataclasses import dataclass
from pathlib import Path

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.ingestion.service import DocumentIngestionService
from agentic_rag.retrieval.service import HybridRetrievalService, RetrievalResult
from agentic_rag.vectorstore.store import LocalVectorStore


@dataclass(frozen=True)
class IndexingSummary:
    """Outcome of indexing one or more source files."""

    source_count: int
    chunk_count: int
    vector_store_path: str


class DocumentIndexingService:
    """Ingest documents, create embeddings, and persist searchable vectors."""

    def __init__(
        self,
        vector_store_path: str | Path,
        ingestion_service: DocumentIngestionService | None = None,
        embedding_provider: LocalHashEmbeddingProvider | None = None,
        semantic_weight: float = 0.7,
    ) -> None:
        self.ingestion_service = ingestion_service or DocumentIngestionService()
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.vector_store = LocalVectorStore(vector_store_path, self.embedding_provider)
        self.semantic_weight = semantic_weight

    def index_file(self, path: str | Path) -> IndexingSummary:
        """Index one file and return the operation summary."""

        source_path = Path(path)
        chunks = self.ingestion_service.ingest_file(source_path)
        self.vector_store.add(chunks)
        return IndexingSummary(1, len(chunks), str(self.vector_store.path))

    def index_directory(self, directory: str | Path) -> IndexingSummary:
        """Index supported files in a directory in stable order."""

        source_directory = Path(directory)
        chunks = self.ingestion_service.ingest_directory(source_directory)
        source_count = sum(
            1
            for path in source_directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}
        )
        self.vector_store.add(chunks)
        return IndexingSummary(source_count, len(chunks), str(self.vector_store.path))

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_name: str | None = None,
        file_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Search indexed content through the hybrid retrieval layer."""

        return HybridRetrievalService(
            self.vector_store, semantic_weight=self.semantic_weight
        ).search(query, top_k, source_name=source_name, file_type=file_type)
