"""Run repeatable quality checks against the assistant workflow."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.workflow.service import KnowledgeAssistantService


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_source: str | None = None
    expected_answer_contains: str | None = None
    expected_answer_keywords: list[str] | None = None
    expected_refusal: bool = False

    def matches_answer(self, answer: str) -> bool:
        """Match meaning-bearing concepts without requiring one exact sentence."""

        normalized_answer = _normalize(answer)
        if self.expected_answer_contains and _normalize(self.expected_answer_contains) not in normalized_answer:
            return False
        if self.expected_answer_keywords:
            return all(
                any(_normalize(option) in normalized_answer for option in keyword.split("|"))
                for keyword in self.expected_answer_keywords
            )
        return True


@dataclass(frozen=True)
class EvaluationReport:
    total_cases: int
    retrieval_hits: int
    answer_matches: int
    citation_hits: int
    correct_refusals: int

    @property
    def retrieval_hit_rate(self) -> float:
        return self._rate(self.retrieval_hits)

    @property
    def answer_match_rate(self) -> float:
        return self._rate(self.answer_matches)

    @property
    def citation_accuracy(self) -> float:
        return self._rate(self.citation_hits)

    @property
    def refusal_accuracy(self) -> float:
        return self._rate(self.correct_refusals)

    def as_dict(self) -> dict[str, int | float]:
        """Return report values in an API- and JSON-friendly shape."""

        return {
            "total_cases": self.total_cases,
            "retrieval_hits": self.retrieval_hits,
            "answer_matches": self.answer_matches,
            "citation_hits": self.citation_hits,
            "correct_refusals": self.correct_refusals,
            "retrieval_hit_rate": round(self.retrieval_hit_rate, 4),
            "answer_match_rate": round(self.answer_match_rate, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "refusal_accuracy": round(self.refusal_accuracy, 4),
        }

    def _rate(self, value: int) -> float:
        return value / self.total_cases if self.total_cases else 0.0


class EvaluationService:
    """Evaluate an assistant against expected source and answer behavior."""

    def __init__(self, assistant: KnowledgeAssistantService) -> None:
        self.assistant = assistant

    @staticmethod
    def load_cases(path: str | Path) -> list[EvaluationCase]:
        """Load evaluation cases from a JSON array."""

        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return [EvaluationCase(**value) for value in values]

    def run(self, cases: list[EvaluationCase]) -> EvaluationReport:
        """Run all cases and calculate simple quality metrics."""

        retrieval_hits = answer_matches = citation_hits = correct_refusals = 0
        for case in cases:
            response = self.assistant.ask(case.question)
            source_names = {citation.source_name for citation in response.citations}
            if case.expected_source is not None and case.expected_source in source_names:
                retrieval_hits += 1
                citation_hits += 1
            if case.matches_answer(response.answer):
                answer_matches += 1
            if response.refusal is case.expected_refusal:
                correct_refusals += 1
        return EvaluationReport(
            total_cases=len(cases),
            retrieval_hits=retrieval_hits,
            answer_matches=answer_matches,
            citation_hits=citation_hits,
            correct_refusals=correct_refusals,
        )


def _normalize(value: str) -> str:
    """Normalize punctuation and common number spellings for flexible checks."""

    normalized = value.lower().replace("thirty", "30").replace("one", "1")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()
