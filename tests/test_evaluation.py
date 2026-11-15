from pathlib import Path

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.evaluation.service import EvaluationService
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.workflow.service import KnowledgeAssistantService

FIXTURES = Path(__file__).parent / "fixtures"
RUNTIME = Path(__file__).parent / ".runtime"


def build_evaluator() -> EvaluationService:
    store_path = RUNTIME / "evaluation.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    service.index_file(FIXTURES / "policy.md")
    assistant = KnowledgeAssistantService(AgentOrchestrator(service), GroundedAnswerService())
    return EvaluationService(assistant)


def test_evaluation_loads_cases_and_calculates_metrics() -> None:
    evaluator = build_evaluator()
    cases = evaluator.load_cases(Path(__file__).parents[1] / "evaluation" / "sample_cases.json")

    report = evaluator.run(cases)

    assert report.total_cases == 3
    assert report.retrieval_hit_rate == 1 / 3
    assert report.answer_match_rate == 2 / 3
    assert report.citation_accuracy == 1 / 3
    assert report.refusal_accuracy == 2 / 3

    values = report.as_dict()
    assert values["retrieval_hit_rate"] == 0.5
    assert values["citation_accuracy"] == 0.5
