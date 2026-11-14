from agentic_rag.config import Settings
from agentic_rag.embeddings.ollama import OllamaEmbeddingProvider
from agentic_rag.embeddings.providers import build_embedding_provider


def test_ollama_embedding_provider_uses_configured_endpoint() -> None:
    provider = build_embedding_provider(
        Settings(
            embedding_provider="ollama",
            embedding_model="test-embedding",
            llm_base_url="http://ollama",
        )
    )

    assert isinstance(provider, OllamaEmbeddingProvider)
    assert provider.model == "test-embedding"
    assert provider.base_url == "http://ollama"


def test_local_embedding_provider_remains_default() -> None:
    provider = build_embedding_provider(Settings(embedding_provider="local"))
    assert provider.__class__.__name__ == "LocalHashEmbeddingProvider"
