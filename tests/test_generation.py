from pathlib import Path

from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


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
