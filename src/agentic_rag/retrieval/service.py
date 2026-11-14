"""Combine semantic and lexical signals into ranked evidence."""

import re
from dataclasses import dataclass

from agentic_rag.ingestion.models import DocumentChunk
from agentic_rag.metrics import metrics
from agentic_rag.vectorstore.store import LocalVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    """Ranked chunk returned by the retrieval layer."""

    chunk: DocumentChunk
    score: float
    semantic_score: float
    lexical_score: float


class HybridRetrievalService:
    """Rank vector-store candidates using semantic and keyword overlap."""

    def __init__(self, vector_store: LocalVectorStore, semantic_weight: float = 0.7) -> None:
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_name: str | None = None,
        file_type: str | None = None,
    ) -> list[RetrievalResult]:
        """Return deduplicated, hybrid-ranked evidence."""

        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        candidates = self.vector_store.search(
            query,
            top_k=max(top_k * 4, 20),
            source_name=source_name,
            file_type=file_type,
        )
        metrics.increment("retrieval_requests_total")
        query_tokens = self._tokens(query)
        results = [
            RetrievalResult(
                chunk=result.chunk,
                score=(self.semantic_weight * result.score)
                + ((1 - self.semantic_weight) * self._lexical_score(query_tokens, result.chunk)),
                semantic_score=result.score,
                lexical_score=self._lexical_score(query_tokens, result.chunk),
            )
            for result in candidates
        ]
        ranked = [
            result
            for result in sorted(results, key=lambda result: result.score, reverse=True)
            if result.score > 0
        ][:top_k]
        metrics.increment("retrieval_results_total", len(ranked))
        return ranked

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[\w'-]+", text.lower()))

    @classmethod
    def _lexical_score(cls, query_tokens: set[str], chunk: DocumentChunk) -> float:
        if not query_tokens:
            return 0.0
        chunk_tokens = cls._tokens(chunk.text)
        return len(query_tokens & chunk_tokens) / len(query_tokens)
