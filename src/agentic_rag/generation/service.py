"""Generate answers that remain tied to retrieved evidence."""

from dataclasses import dataclass

from agentic_rag.retrieval.service import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """A source reference attached to grounded answer evidence."""

    citation_id: int
    chunk_id: str
    source_name: str
    page_number: int | None


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer text plus machine-readable citations and grounding status."""

    answer: str
    citations: list[Citation]
    grounded: bool
    refusal: bool


class GroundedAnswerService:
    """Create a deterministic extractive answer from retrieved evidence."""

    def generate(self, question: str, results: list[RetrievalResult]) -> GroundedAnswer:
        """Generate an answer without adding facts absent from the evidence."""

        if not question.strip():
            raise ValueError("question must not be empty")
        if not results:
            return GroundedAnswer(
                answer=(
                    "I could not find enough reliable information in the indexed documents "
                    "to answer that question confidently."
                ),
                citations=[],
                grounded=False,
                refusal=True,
            )

        citations = [
            Citation(
                citation_id=index,
                chunk_id=result.chunk.chunk_id,
                source_name=result.chunk.source_name,
                page_number=result.chunk.page_number,
            )
            for index, result in enumerate(results, start=1)
        ]
        evidence_lines = [
            f"[{citation.citation_id}] {result.chunk.text.strip()}"
            for citation, result in zip(citations, results, strict=True)
        ]
        return GroundedAnswer(
            answer="Based on the indexed sources:\n" + "\n".join(evidence_lines),
            citations=citations,
            grounded=True,
            refusal=False,
        )
