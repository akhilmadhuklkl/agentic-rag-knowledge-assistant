"""Optional local language-model providers."""

import httpx

from agentic_rag.config import Settings
from agentic_rag.reliability import retry_call


class OllamaAnswerModel:
    """Generate answers through a locally running Ollama instance."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def generate(self, prompt: str) -> str:
        """Call Ollama's local generation endpoint."""

        def request() -> httpx.Response:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response

        response = retry_call(
            request,
            retry_exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            max_attempts=self.max_retries + 1,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("response", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty answer")
        return answer


def build_answer_model(settings: Settings):
    """Build the configured provider; deterministic mode uses no external service."""

    if settings.llm_provider.lower() == "ollama":
        return OllamaAnswerModel(
            settings.llm_model,
            settings.llm_base_url,
            max_retries=settings.llm_max_retries,
        )
    return None
