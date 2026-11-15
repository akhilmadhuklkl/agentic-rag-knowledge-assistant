"""Coordinate planning, retrieval, generation, and verification."""

from dataclasses import dataclass
from time import perf_counter

from agentic_rag.agents.models import PlanAction
from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.generation.service import Citation, GroundedAnswerService
from agentic_rag.metrics import metrics
from agentic_rag.observability import log_event
from agentic_rag.verification.service import AnswerVerifier


@dataclass(frozen=True)
class AssistantResponse:
    """Complete response returned by the assistant workflow."""

    answer: str
    citations: list[Citation]
    grounded: bool
    refusal: bool
    verified: bool
    verification_reason: str
    action: PlanAction
    attempts: int
    duration_ms: float


class KnowledgeAssistantService:
    """Run the full bounded, citation-aware assistant workflow."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        answer_service: GroundedAnswerService | None = None,
        verifier: AnswerVerifier | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.answer_service = answer_service or GroundedAnswerService()
        self.verifier = verifier or AnswerVerifier()

    def ask(self, question: str, top_k: int = 5, semantic_weight: float | None = None, file_type: str | None = None) -> AssistantResponse:
        """Answer one question through the complete workflow."""

        started = perf_counter()
        orchestration = self.orchestrator.run(question, top_k=top_k, semantic_weight=semantic_weight, file_type=file_type)
        generated = self.answer_service.generate(question, orchestration.results)
        verification = self.verifier.verify(generated, orchestration.results)
        duration_ms = round((perf_counter() - started) * 1000, 2)
        log_event(
            "assistant_workflow_completed",
            action=orchestration.plan.action.value,
            attempts=orchestration.attempts,
            grounded=generated.grounded,
            verified=verification.verified,
            duration_ms=duration_ms,
        )
        metrics.increment("assistant_requests_total")
        metrics.increment("assistant_refusals_total" if generated.refusal else "assistant_answers_total")
        metrics.increment("assistant_verified_total" if verification.verified else "assistant_unverified_total")
        metrics.observe("assistant_workflow_duration_ms", duration_ms)
        return AssistantResponse(
            answer=generated.answer,
            citations=generated.citations,
            grounded=generated.grounded,
            refusal=generated.refusal,
            verified=verification.verified,
            verification_reason=verification.reason,
            action=orchestration.plan.action,
            attempts=orchestration.attempts,
            duration_ms=duration_ms,
        )
