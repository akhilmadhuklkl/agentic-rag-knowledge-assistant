from pathlib import Path

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.workflow.service import KnowledgeAssistantService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def build_assistant() -> KnowledgeAssistantService:
    store_path = RUNTIME / "workflow.json"
    store_path.unlink(missing_ok=True)
    indexing = DocumentIndexingService(store_path)
    indexing.index_file(FIXTURES / "policy.md")
    return KnowledgeAssistantService(AgentOrchestrator(indexing), GroundedAnswerService())


def test_end_to_end_workflow_returns_verified_cited_answer() -> None:
    response = build_assistant().ask("What is the refund policy?", top_k=1)

    assert response.grounded is True
    assert response.verified is True
    assert response.refusal is False
    assert response.citations[0].source_name == "policy.md"


def test_end_to_end_workflow_returns_verified_refusal() -> None:
    response = build_assistant().ask("What is the quantum spaceship launch protocol?", top_k=1)

    assert response.refusal is True
    assert response.verified is True
    assert response.citations == []
