from pathlib import Path

from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


class FakeAnswerModel:
    def generate(self, prompt: str) -> str:
        assert "Question:" in prompt
        assert "Evidence:" in prompt
        return "The policy allows refunds within thirty days."


def test_grounded_answer_contains_evidence_and_citations() -> None:
    store_path = RUNTIME / "generation.json"
    store_path.unlink(missing_ok=True)
    indexing = DocumentIndexingService(store_path)
    indexing.index_file(FIXTURES / "policy.md")

    answer = GroundedAnswerService().generate(
        "What is the refund policy?", indexing.search("refund policy")
    )

    assert answer.grounded is True
    assert answer.refusal is False
    assert "Refunds are available" in answer.answer
    assert answer.citations[0].source_name == "policy.md"


def test_grounded_answer_refuses_without_evidence() -> None:
    answer = GroundedAnswerService().generate("Unknown question", [])

    assert answer.grounded is False
    assert answer.refusal is True
    assert answer.citations == []


def test_grounded_answer_can_use_a_model_without_owning_citations() -> None:
    store_path = RUNTIME / "generation-model.json"
    store_path.unlink(missing_ok=True)
    indexing = DocumentIndexingService(store_path)
    indexing.index_file(FIXTURES / "policy.md")

    answer = GroundedAnswerService(FakeAnswerModel()).generate(
        "What is the refund policy?", indexing.search("refund policy")
    )

    assert answer.answer == "The policy allows refunds within thirty days."
    assert answer.citations[0].source_name == "policy.md"
