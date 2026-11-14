from pathlib import Path

from agentic_rag.api.routes import ask, conversation_store
from agentic_rag.api.schemas import AnswerRequest
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def test_ask_route_returns_and_reuses_conversation_id() -> None:
    store_path = RUNTIME / "conversation-api.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_file(FIXTURES / "policy.md")

    first = ask(AnswerRequest(query="What is the refund policy?"), service)
    second = ask(
        AnswerRequest(query="What about annual plans?", conversation_id=first.conversation_id),
        service,
    )

    assert first.conversation_id == second.conversation_id
    assert len(conversation_store.get(first.conversation_id)) >= 2
