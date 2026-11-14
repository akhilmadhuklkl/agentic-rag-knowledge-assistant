"""Embedding provider abstractions and local implementations."""

from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.embeddings.ollama import OllamaEmbeddingProvider

__all__ = ["LocalHashEmbeddingProvider", "OllamaEmbeddingProvider"]
