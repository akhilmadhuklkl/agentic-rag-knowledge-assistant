from agentic_rag.config import Settings
from agentic_rag.generation.providers import OllamaAnswerModel, build_answer_model


def test_deterministic_mode_does_not_require_external_service() -> None:
    assert build_answer_model(Settings(llm_provider="deterministic")) is None


def test_ollama_provider_uses_configured_endpoint() -> None:
    provider = build_answer_model(
        Settings(llm_provider="ollama", llm_model="test-model", llm_base_url="http://ollama")
    )

    assert isinstance(provider, OllamaAnswerModel)
    assert provider.model == "test-model"
    assert provider.base_url == "http://ollama"
