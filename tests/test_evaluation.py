from pathlib import Path

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.evaluation.service import EvaluationCase, EvaluationService
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


def build_multi_document_evaluator() -> EvaluationService:
    store_path = RUNTIME / "multi-document-evaluation.json"
    store_path.unlink(missing_ok=True)
    service = DocumentIndexingService(store_path)
    fixtures = Path(__file__).parents[1] / "evaluation" / "fixtures"
    for document in sorted(fixtures.iterdir()):
        service.index_file(document)
    assistant = KnowledgeAssistantService(AgentOrchestrator(service), GroundedAnswerService())
    return EvaluationService(assistant)


def test_evaluation_loads_cases_and_calculates_metrics() -> None:
    evaluator = build_evaluator()
    cases = evaluator.load_cases(Path(__file__).parents[1] / "evaluation" / "sample_cases.json")

    report = evaluator.run(cases)

    assert report.total_cases == 3
    assert report.retrieval_hit_rate == 2 / 3
    assert report.answer_match_rate == 1.0
    assert report.citation_accuracy == 2 / 3
    assert report.refusal_accuracy == 1.0

    values = report.as_dict()
    assert values["retrieval_hit_rate"] == 0.6667
    assert values["citation_accuracy"] == 0.6667


def test_evaluation_accepts_equivalent_answer_wording() -> None:
    case = EvaluationCase(
        question="What is the refund policy?",
        expected_answer_keywords=["refund", "30|thirty", "days"],
    )

    assert case.matches_answer("Refund requests are accepted within 30 days.")
    assert case.matches_answer("The refund policy allows requests within thirty days.")
    assert not case.matches_answer("Refund requests are accepted within 14 days.")


def test_multi_document_benchmark_selects_sources_and_refuses_unknown_topics() -> None:
    evaluator = build_multi_document_evaluator()
    cases = evaluator.load_cases(
        Path(__file__).parents[1] / "evaluation" / "multi_document_cases.json"
    )

    report = evaluator.run(cases)

    assert report.total_cases == 6
    assert report.retrieval_hits == 5
    assert report.answer_matches == 6
    assert report.citation_hits == 5
    assert report.correct_refusals == 6
