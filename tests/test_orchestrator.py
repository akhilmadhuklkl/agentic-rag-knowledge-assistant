from pathlib import Path

from agentic_rag.agents.models import PlanAction
from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.indexing.service import DocumentIndexingService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def build_orchestrator() -> AgentOrchestrator:
    store_path = RUNTIME / "orchestrator.json"
    store_path.unlink(missing_ok=True)
    store_path.with_suffix(".manifest.json").unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_file(FIXTURES / "policy.md")
    return AgentOrchestrator(service)


def test_orchestrator_plans_document_search() -> None:
    plan = build_orchestrator().plan("What is the refund policy?")

    assert plan.action is PlanAction.SEARCH_DOCUMENTS
    assert plan.max_attempts == 2


def test_orchestrator_runs_search_and_returns_evidence() -> None:
    result = build_orchestrator().run("What is the refund policy?", top_k=1)

    assert result.attempts == 1
    assert result.needs_clarification is False
    assert result.results[0].chunk.source_name == "policy.md"


def test_orchestrator_retries_when_first_search_has_no_evidence() -> None:
    result = build_orchestrator().run("What are vacation rules?", top_k=1)

    assert result.attempts == 2
    assert result.needs_clarification is True
    assert result.results == []


def test_orchestrator_clarifies_short_questions() -> None:
    result = build_orchestrator().run("Refund?")

    assert result.plan.action is PlanAction.CLARIFY
    assert result.needs_clarification is True
    assert result.attempts == 0
