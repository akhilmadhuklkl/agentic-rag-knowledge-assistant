import json

import pytest

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
