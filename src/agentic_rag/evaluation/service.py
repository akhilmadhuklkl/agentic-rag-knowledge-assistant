"""Run repeatable quality checks against the assistant workflow."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.workflow.service import KnowledgeAssistantService


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    category: str = "general"
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
    cases: list["EvaluationCaseResult"]
    answerable_cases: int
    refusal_cases: int
    refusal_case_correct: int
    category_metrics: dict[str, dict[str, int | float]]

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

    @property
    def answerable_retrieval_rate(self) -> float:
        return self.retrieval_hits / self.answerable_cases if self.answerable_cases else 0.0

    @property
    def answerable_citation_accuracy(self) -> float:
        return self.citation_hits / self.answerable_cases if self.answerable_cases else 0.0

    @property
    def refusal_case_accuracy(self) -> float:
        return self.refusal_case_correct / self.refusal_cases if self.refusal_cases else 0.0

    def as_dict(self) -> dict[str, int | float | list[dict[str, object]]]:
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
            "answerable_cases": self.answerable_cases,
            "refusal_cases": self.refusal_cases,
            "answerable_retrieval_rate": round(self.answerable_retrieval_rate, 4),
            "answerable_citation_accuracy": round(self.answerable_citation_accuracy, 4),
            "refusal_case_accuracy": round(self.refusal_case_accuracy, 4),
            "refusal_case_correct": self.refusal_case_correct,
            "category_metrics": self.category_metrics,
            "cases": [case.as_dict() for case in self.cases],
        }

    def _rate(self, value: int) -> float:
        return value / self.total_cases if self.total_cases else 0.0


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Explain the outcome of one benchmark question."""

    question: str
    category: str
    expected_source: str | None
    cited_sources: list[str]
    retrieval_hit: bool
    answer_match: bool
    citation_hit: bool
    refusal_correct: bool

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


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

        retrieval_hits = answer_matches = citation_hits = correct_refusals = refusal_case_correct = 0
        category_totals: dict[str, dict[str, int]] = {}
        answerable_cases = sum(case.expected_source is not None for case in cases)
        refusal_cases = sum(case.expected_refusal for case in cases)
        case_results: list[EvaluationCaseResult] = []
        for case in cases:
            category_totals.setdefault(case.category, {"cases": 0, "retrieval_hits": 0, "answer_matches": 0, "refusal_cases": 0, "refusal_correct": 0})
            category = category_totals[case.category]
            category["cases"] += 1
            category["refusal_cases"] += int(case.expected_refusal)
            response = self.assistant.ask(case.question)
            source_names = sorted({citation.source_name for citation in response.citations})
            retrieval_hit = case.expected_source is not None and case.expected_source in source_names
            citation_hit = retrieval_hit
            answer_match = case.matches_answer(response.answer)
            refusal_correct = response.refusal is case.expected_refusal
            if retrieval_hit:
                retrieval_hits += 1
                category["retrieval_hits"] += 1
            if citation_hit:
                citation_hits += 1
            if answer_match:
                answer_matches += 1
                category["answer_matches"] += 1
            if refusal_correct:
                correct_refusals += 1
                if case.expected_refusal:
                    refusal_case_correct += 1
                    category["refusal_correct"] += 1
            case_results.append(EvaluationCaseResult(case.question, case.category, case.expected_source, source_names, retrieval_hit, answer_match, citation_hit, refusal_correct))
        category_metrics = {
            name: {
                **values,
                "retrieval_rate": round(values["retrieval_hits"] / values["cases"], 4),
                "answer_rate": round(values["answer_matches"] / values["cases"], 4),
                "refusal_rate": round(values["refusal_correct"] / values["refusal_cases"], 4) if values["refusal_cases"] else 0.0,
            }
            for name, values in category_totals.items()
        }
        return EvaluationReport(
            total_cases=len(cases),
            retrieval_hits=retrieval_hits,
            answer_matches=answer_matches,
            citation_hits=citation_hits,
            correct_refusals=correct_refusals,
            cases=case_results,
            answerable_cases=answerable_cases,
            refusal_cases=refusal_cases,
            refusal_case_correct=refusal_case_correct,
            category_metrics=category_metrics,
        )


def _normalize(value: str) -> str:
    """Normalize punctuation and common number spellings for flexible checks."""

    normalized = value.lower().replace("thirty", "30").replace("one", "1")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()
