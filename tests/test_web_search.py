import json

import pytest

from agentic_rag.generation.service import GroundedAnswerService
from agentic_rag.ingestion.models import DocumentChunk
from agentic_rag.retrieval.service import RetrievalResult
from agentic_rag.tools.web_search import WebSearchTool


def test_web_search_rejects_empty_query() -> None:
    with pytest.raises(ValueError, match="query"):
        WebSearchTool().search(" ")


def test_web_search_returns_bounded_results(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"RelatedTopics": [{"Text": f"Result {i}", "FirstURL": f"https://example.com/{i}"} for i in range(5)]}).encode()

    monkeypatch.setattr("agentic_rag.tools.web_search.urlopen", lambda request, timeout: Response())

    results = WebSearchTool(max_results=2).search("atlas")

    assert len(results) == 2
    assert results[0].url == "https://example.com/0"


def test_external_result_preserves_clickable_source_url() -> None:
    chunk = DocumentChunk("web:1", "External evidence", "https://example.com", "Example source", "web", None, 1, "standard", "current")
    result = RetrievalResult(chunk, 0.5, 0.0, 0.0, file_type="web")

    citation = GroundedAnswerService().generate("What is this?", [result]).citations[0]

    assert citation.source_url == "https://example.com"
    assert citation.file_type == "web"


def test_web_search_fails_closed_on_network_errors(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("agentic_rag.tools.web_search.urlopen", fail)

    assert WebSearchTool().search("offline question") == []
