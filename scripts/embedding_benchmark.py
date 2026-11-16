"""Compare retrieval quality across embedding providers."""

import argparse
import json
import tempfile
from pathlib import Path

from agentic_rag.agents.orchestrator import AgentOrchestrator
from agentic_rag.embeddings.local import LocalHashEmbeddingProvider
from agentic_rag.embeddings.ollama import OllamaEmbeddingProvider
from agentic_rag.evaluation.service import EvaluationService
from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.workflow.service import KnowledgeAssistantService

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "evaluation" / "fixtures"
CASES = ROOT / "evaluation" / "multi_document_cases.json"


def build_provider(name: str, model: str, base_url: str):
    """Create the requested embedding provider."""

    if name == "ollama":
        return OllamaEmbeddingProvider(model=model, base_url=base_url)
    return LocalHashEmbeddingProvider()


def run(provider_name: str, model: str, base_url: str) -> dict[str, object]:
    """Index the benchmark corpus and return quality metrics."""

    with tempfile.TemporaryDirectory(prefix="rag-embedding-benchmark-") as directory:
        indexer = DocumentIndexingService(
            Path(directory) / "vectors.json",
            embedding_provider=build_provider(provider_name, model, base_url),
        )
        indexer.index_directory(FIXTURES)
        assistant = KnowledgeAssistantService(
            AgentOrchestrator(indexer), GroundedAnswerService()
        )
        report = EvaluationService(assistant).run(EvaluationService.load_cases(CASES))
        return {
            "provider": provider_name,
            "model": model if provider_name == "ollama" else None,
            "total_cases": report.total_cases,
            "answerable_cases": report.answerable_cases,
            "answerable_retrieval_rate": round(report.answerable_retrieval_rate, 4),
            "answerable_citation_accuracy": round(report.answerable_citation_accuracy, 4),
            "answer_match_rate": round(report.answer_match_rate, 4),
            "refusal_case_accuracy": round(report.refusal_case_accuracy, 4),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("local", "ollama"), default="local")
    parser.add_argument("--model", default="nomic-embed-text")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = run(args.provider, args.model, args.base_url)
    print(json.dumps(result, indent=2) if args.as_json else _format(result))


def _format(result: dict[str, object]) -> str:
    """Render a compact human-readable benchmark summary."""

    return (
        f"{result['provider']} embeddings: "
        f"retrieval={result['answerable_retrieval_rate']:.0%}, "
        f"citations={result['answerable_citation_accuracy']:.0%}, "
        f"answers={result['answer_match_rate']:.0%}, "
        f"refusals={result['refusal_case_accuracy']:.0%}"
    )


if __name__ == "__main__":
    main()
