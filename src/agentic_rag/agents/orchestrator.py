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
        minimum_evidence_score: float = 0.4,
        minimum_lexical_overlap: float = 0.3,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if not 0 <= minimum_evidence_score <= 1:
            raise ValueError("minimum_evidence_score must be between 0 and 1")
        if not 0 <= minimum_lexical_overlap <= 1:
            raise ValueError("minimum_lexical_overlap must be between 0 and 1")
        self.indexing_service = indexing_service
        self.max_attempts = max_attempts
        self.minimum_evidence_score = minimum_evidence_score
        self.minimum_lexical_overlap = minimum_lexical_overlap

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

    def run(self, question: str, top_k: int = 5, semantic_weight: float | None = None, file_type: str | None = None) -> OrchestrationResult:
        """Execute the plan with at most ``max_attempts`` retrieval calls."""

        plan = self.plan(question)
        if plan.action is PlanAction.CLARIFY:
            return OrchestrationResult(plan, [], 0, True)

        results = self._reliable_results(plan.query, top_k, semantic_weight, file_type)
        if " and " in plan.query.lower():
            results = self._merge_multi_query_results(plan.query, results, top_k, semantic_weight, file_type)
        attempts = 1
        if not results and attempts < plan.max_attempts:
            refined_query = self._refine_query(plan.query)
            results = self._reliable_results(refined_query, top_k, semantic_weight, file_type)
            attempts += 1
        return OrchestrationResult(plan, results, attempts, not results)

    def _merge_multi_query_results(
        self,
        question: str,
        initial: list[RetrievalResult],
        top_k: int,
        semantic_weight: float | None,
        file_type: str | None,
    ) -> list[RetrievalResult]:
        """Collect evidence for each conjunction clause and remove duplicates."""

        clauses = [part.strip(" ,") for part in question.split(" and ") if len(part.split()) >= 3]
        merged = {result.chunk.chunk_id: result for result in initial}
        for clause in clauses:
            clause_results = self.indexing_service.search(clause, top_k=2, semantic_weight=semantic_weight, file_type=file_type)
            for result in clause_results:
                merged.setdefault(result.chunk.chunk_id, result)
        return sorted(merged.values(), key=lambda result: result.score, reverse=True)[:top_k]

    def _reliable_results(self, query: str, top_k: int, semantic_weight: float | None = None, file_type: str | None = None) -> list[RetrievalResult]:
        results = self.indexing_service.search(query, top_k=top_k, semantic_weight=semantic_weight, file_type=file_type)
        return [
            result
            for result in results
            if (
                result.score >= self.minimum_evidence_score
                and result.lexical_score >= self.minimum_lexical_overlap
            )
            or (
                result.lexical_score >= 0.3
                and result.semantic_score >= 0.25
            )
            or (
                result.score >= self.minimum_evidence_score
                and result.semantic_score >= 0.35
            )
        ]

    @staticmethod
    def _refine_query(question: str) -> str:
        """Use a compact keyword-focused retry query."""

        stop_words = {
            "what", "when", "where", "which", "how", "does", "do", "is", "are", "the",
            "i", "have", "to", "a",
        }
        words = [
            word for word in question.split() if word.lower().strip("?!.,") not in stop_words
        ]
        normalized = {word.lower().strip("?!.,") for word in words}
        expansions = {"refund": "policy", "refunded": "policy", "reimbursement": "policy"}
        words.extend(term for source, term in expansions.items() if source in normalized)
        return " ".join(words)
