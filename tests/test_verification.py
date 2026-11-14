from pathlib import Path

from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.verification.service import AnswerVerifier

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def test_verifier_accepts_citations_from_retrieved_evidence() -> None:
    store_path = RUNTIME / "verification.json"
    store_path.unlink(missing_ok=True)
    indexing = DocumentIndexingService(store_path)
    indexing.index_file(FIXTURES / "policy.md")
    evidence = indexing.search("refund policy")
    answer = GroundedAnswerService().generate("refund policy", evidence)

    result = AnswerVerifier().verify(answer, evidence)

    assert result.verified is True


def test_verifier_accepts_a_safe_refusal() -> None:
    answer = GroundedAnswerService().generate("unknown", [])

    assert AnswerVerifier().verify(answer, []).verified is True
