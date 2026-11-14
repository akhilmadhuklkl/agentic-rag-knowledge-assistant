"""Validate that generated answers remain connected to retrieved evidence."""

from dataclasses import dataclass

from agentic_rag.generation.service import GroundedAnswer
from agentic_rag.retrieval.service import RetrievalResult


@dataclass(frozen=True)
class AnswerVerification:
    """Verification outcome for a generated answer."""

    verified: bool
    reason: str


class AnswerVerifier:
    """Perform deterministic structural grounding checks."""

    def verify(
        self,
        answer: GroundedAnswer,
        evidence: list[RetrievalResult],
    ) -> AnswerVerification:
        """Check answer presence and ensure citations map to retrieved chunks."""

        if answer.refusal:
            return AnswerVerification(True, "The assistant refused because evidence was insufficient.")
        if not answer.answer.strip():
            return AnswerVerification(False, "The generated answer is empty.")
        evidence_ids = {result.chunk.chunk_id for result in evidence}
        invalid = [citation for citation in answer.citations if citation.chunk_id not in evidence_ids]
        if invalid:
            return AnswerVerification(False, "One or more citations do not match retrieved evidence.")
        if not answer.citations:
            return AnswerVerification(False, "A non-refusal answer must include citations.")
        return AnswerVerification(True, "All citations map to retrieved evidence.")
