"""Application service connecting document ingestion to vector storage."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.ingestion.service import DocumentIngestionService
from agentic_rag.metrics import metrics
from agentic_rag.retrieval.service import HybridRetrievalService, RetrievalResult
from agentic_rag.storage.documents import StoredDocument
from agentic_rag.vectorstore.store import LocalVectorStore


@dataclass(frozen=True)
class IndexingSummary:
    """Outcome of indexing one or more source files."""

    source_count: int
    chunk_count: int
    vector_store_path: str
    indexed_count: int = 0
    skipped_count: int = 0


class DocumentIndexingService:
    """Ingest documents, create embeddings, and persist searchable vectors."""

    def __init__(
        self,
        vector_store_path: str | Path,
        ingestion_service: DocumentIngestionService | None = None,
        embedding_provider: Any | None = None,
        semantic_weight: float = 0.7,
        metadata_repository: Any | None = None,
    ) -> None:
        self.ingestion_service = ingestion_service or DocumentIngestionService()
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.vector_store = LocalVectorStore(vector_store_path, self.embedding_provider)
        self.semantic_weight = semantic_weight
        self.metadata_repository = metadata_repository
        self.manifest_path = Path(vector_store_path).with_suffix(".manifest.json")
        self._manifest = self._load_manifest()

    def index_file(self, path: str | Path, force: bool = False) -> IndexingSummary:
        """Index one file and return the operation summary."""

        source_path = Path(path)
        fingerprint = self._fingerprint(source_path)
        known_fingerprint = (
            self.metadata_repository.get_fingerprint(str(source_path))
            if self.metadata_repository
            else self._manifest.get(str(source_path))
        )
        if not force and known_fingerprint == fingerprint:
            metrics.increment("documents_index_skipped_total")
            return IndexingSummary(1, 0, str(self.vector_store.path), skipped_count=1)
        chunks = self.ingestion_service.ingest_file(source_path)
        if self.metadata_repository:
            self.metadata_repository.upsert_document(
                StoredDocument(
                    name=source_path.name,
                    path=str(source_path),
                    size_bytes=source_path.stat().st_size,
                    file_type=source_path.suffix.lower().lstrip("."),
                )
            )
        self.vector_store.delete_source(str(source_path))
        self.vector_store.add(chunks)
        if self.metadata_repository:
            self.metadata_repository.set_fingerprint(str(source_path), fingerprint)
        else:
            self._manifest[str(source_path)] = fingerprint
            self._save_manifest()
        metrics.increment("documents_indexed_total")
        metrics.increment("chunks_indexed_total", len(chunks))
        return IndexingSummary(1, len(chunks), str(self.vector_store.path), indexed_count=1)

    def reindex_file(self, path: str | Path) -> IndexingSummary:
        """Force a source to be re-read and re-embedded."""

        return self.index_file(path, force=True)

    def cleanup_missing_sources(self, directory: str | Path) -> int:
        """Remove indexed sources that no longer exist under a directory."""

        root = Path(directory).resolve()
        removed = 0
        for source_path in list(self.vector_store.source_paths()):
            path = Path(source_path)
            if path.is_relative_to(root) and not path.is_file():
                self.vector_store.delete_source(source_path)
                if self.metadata_repository:
                    self.metadata_repository.delete_fingerprint(source_path)
                else:
                    self._manifest.pop(source_path, None)
                removed += 1
        if not self.metadata_repository:
            self._save_manifest()
        return removed

    def index_directory(self, directory: str | Path) -> IndexingSummary:
        """Index supported files in a directory in stable order."""

        source_directory = Path(directory)
        total = IndexingSummary(0, 0, str(self.vector_store.path))
        for path in sorted(source_directory.iterdir()):
            if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
                summary = self.index_file(path)
                total = IndexingSummary(
                    total.source_count + summary.source_count,
                    total.chunk_count + summary.chunk_count,
                    total.vector_store_path,
                    total.indexed_count + summary.indexed_count,
                    total.skipped_count + summary.skipped_count,
                )
        return total

    @staticmethod
    def _fingerprint(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _load_manifest(self) -> dict[str, str]:
        if not self.manifest_path.is_file() or not self.vector_store.path.is_file():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")

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
