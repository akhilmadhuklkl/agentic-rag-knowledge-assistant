"""Models used by the agent orchestration workflow."""

from dataclasses import dataclass
from enum import StrEnum

from agentic_rag.retrieval.service import RetrievalResult


class PlanAction(StrEnum):
    """Actions currently understood by the deterministic planner."""

    SEARCH_DOCUMENTS = "search_documents"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class QueryPlan:
    """A bounded plan produced before tools are executed."""

    action: PlanAction
    query: str
    reason: str
    max_attempts: int = 2


@dataclass(frozen=True)
class OrchestrationResult:
    """Evidence and execution details returned by the orchestrator."""

    plan: QueryPlan
    results: list[RetrievalResult]
    attempts: int
    needs_clarification: bool
