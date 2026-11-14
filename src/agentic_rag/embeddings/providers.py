"""Factory for configured embedding providers."""

from agentic_rag.config import Settings
from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.embeddings.ollama import OllamaEmbeddingProvider


def build_embedding_provider(settings: Settings):
    """Build the configured provider; local mode requires no external service."""

    if settings.embedding_provider.lower() == "ollama":
        return OllamaEmbeddingProvider(
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
            max_retries=settings.llm_max_retries,
        )
    return LocalHashEmbeddingProvider()
