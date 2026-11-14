"""Ollama-backed embedding provider."""

import httpx

from agentic_rag.reliability import retry_call


class OllamaEmbeddingProvider:
    """Create embeddings through a locally running Ollama instance."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def provider_id(self) -> str:
        """Stable identifier stored with vector indexes."""

        return f"ollama:{self.model}"

    def embed(self, text: str) -> list[float]:
        """Create one embedding vector."""

        def request() -> httpx.Response:
            response = httpx.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = retry_call(
            request,
            retry_exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            max_attempts=self.max_retries + 1,
        )
        vector = response.json().get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Ollama returned an empty embedding")
        return [float(value) for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Create vectors in input order."""

        return [self.embed(text) for text in texts]
