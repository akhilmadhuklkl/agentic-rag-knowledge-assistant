"""Coordinate planning, retrieval, generation, and verification."""

from dataclasses import dataclass

from agentic_rag.agents.models import PlanAction
from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.generation.service import Citation, GroundedAnswerService
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

    def ask(self, question: str, top_k: int = 5) -> AssistantResponse:
        """Answer one question through the complete workflow."""

        orchestration = self.orchestrator.run(question, top_k=top_k)
        generated = self.answer_service.generate(question, orchestration.results)
        verification = self.verifier.verify(generated, orchestration.results)
        return AssistantResponse(
            answer=generated.answer,
            citations=generated.citations,
            grounded=generated.grounded,
            refusal=generated.refusal,
            verified=verification.verified,
            verification_reason=verification.reason,
            action=orchestration.plan.action,
            attempts=orchestration.attempts,
        )
