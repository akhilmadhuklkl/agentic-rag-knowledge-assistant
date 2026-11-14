"""Deterministic agent orchestration for bounded document research."""

from agentic_rag.agents.models import OrchestrationResult, PlanAction, QueryPlan
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.retrieval.service import RetrievalResult


class AgentOrchestrator:
    """Plan and execute document research without unbounded tool calls."""

    def __init__(
        self,
        indexing_service: DocumentIndexingService,
        max_attempts: int = 2,
        minimum_evidence_score: float = 0.25,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not 0 <= minimum_evidence_score <= 1:
            raise ValueError("minimum_evidence_score must be between 0 and 1")
        self.indexing_service = indexing_service
        self.max_attempts = max_attempts
        self.minimum_evidence_score = minimum_evidence_score

    def plan(self, question: str) -> QueryPlan:
        """Create a simple plan from the user question."""

        normalized = question.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        if len(normalized.split()) < 2:
            return QueryPlan(
                action=PlanAction.CLARIFY,
                query=normalized,
                reason="The question is too short to search reliably.",
                max_attempts=self.max_attempts,
            )
        return QueryPlan(
            action=PlanAction.SEARCH_DOCUMENTS,
            query=normalized,
            reason="The question may be answered from indexed documents.",
            max_attempts=self.max_attempts,
        )

    def run(self, question: str, top_k: int = 5) -> OrchestrationResult:
        """Execute the plan with at most ``max_attempts`` retrieval calls."""

        plan = self.plan(question)
        if plan.action is PlanAction.CLARIFY:
            return OrchestrationResult(plan, [], 0, True)

        results = self._reliable_results(plan.query, top_k)
        attempts = 1
        if not results and attempts < plan.max_attempts:
            refined_query = self._refine_query(plan.query)
            results = self._reliable_results(refined_query, top_k)
            attempts += 1
        return OrchestrationResult(plan, results, attempts, not results)

    def _reliable_results(self, query: str, top_k: int) -> list[RetrievalResult]:
        results = self.indexing_service.search(query, top_k=top_k)
        return [result for result in results if result.score >= self.minimum_evidence_score]

    @staticmethod
    def _refine_query(question: str) -> str:
        """Use a compact keyword-focused retry query."""

        stop_words = {"what", "when", "where", "which", "how", "does", "is", "are", "the"}
        return " ".join(word for word in question.split() if word.lower().strip("?!.,") not in stop_words)
