"""A small persistent vector store for the local MVP."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.ingestion.models import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    """A document chunk matched against a query vector."""

    chunk: DocumentChunk
    score: float


class LocalVectorStore:
    """Persist chunk embeddings as JSON and search them by cosine similarity."""

    def __init__(self, path: str | Path, embedding_provider: LocalHashEmbeddingProvider) -> None:
        self.path = Path(path)
        self.embedding_provider = embedding_provider
        self.metadata_path = self.path.with_suffix(".meta.json")
        self.provider_id = getattr(embedding_provider, "provider_id", type(embedding_provider).__name__)
        self._records: list[dict[str, object]] = []
        self._load()

    def add(self, chunks: list[DocumentChunk]) -> None:
        """Upsert chunks by chunk ID and persist their embeddings."""

        existing = {str(record["chunk"]["chunk_id"]): record for record in self._records}
        vectors = self.embedding_provider.embed_many([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
            existing[chunk.chunk_id] = {"chunk": asdict(chunk), "vector": vector}
        self._records = list(existing.values())
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_name: str | None = None,
        file_type: str | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks for a query."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_vector = self.embedding_provider.embed(query)
        scored = [
            SearchResult(
                chunk=DocumentChunk(**record["chunk"]),
                score=self._cosine(query_vector, record["vector"]),
            )
            for record in self._records
            if (source_name is None or record["chunk"]["source_name"] == source_name)
            and (
                file_type is None
                or record["chunk"]["file_type"] == file_type.lower().lstrip(".")
            )
        ]
        return sorted(scored, key=lambda result: result.score, reverse=True)[:top_k]

    def count(self) -> int:
        """Return the number of stored chunks."""

        return len(self._records)

    def delete_source(self, source_path: str) -> None:
        """Remove all chunks belonging to one source before re-indexing it."""

        retained = [
            record for record in self._records if record["chunk"]["source_path"] != source_path
        ]
        if len(retained) != len(self._records):
            self._records = retained
            self._save()

    def source_paths(self) -> set[str]:
        """Return source paths represented in the vector store."""

        return {str(record["chunk"]["source_path"]) for record in self._records}

    def _load(self) -> None:
        if self.path.is_file():
            self._records = json.loads(self.path.read_text(encoding="utf-8"))
        if self.metadata_path.is_file():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("provider_id") != self.provider_id:
                raise ValueError(
                    "embedding provider does not match the existing vector index; "
                    "re-index documents before switching providers"
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        self.metadata_path.write_text(
            json.dumps({"provider_id": self.provider_id}, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("vectors must have equal dimensions")
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if left_magnitude == 0 or right_magnitude == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            left_magnitude * right_magnitude
        )
