from pathlib import Path

from agentic_rag.api.routes import answer
from agentic_rag.api.schemas import AnswerRequest
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def test_answer_route_returns_grounded_response() -> None:
    store_path = RUNTIME / "api-answer.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_file(FIXTURES / "policy.md")

    response = answer(AnswerRequest(query="What is the refund policy?"), service)

    assert response.grounded is True
    assert response.refusal is False
    assert response.citations[0].source_name == "policy.md"
