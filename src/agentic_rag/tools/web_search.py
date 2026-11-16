"""Bounded web-search tool kept separate from trusted document retrieval."""

import json
from dataclasses import dataclass
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class WebSearchResult:
    """A compact external search result that can be cited separately."""

    title: str
    url: str
    snippet: str


class WebSearchTool:
    """Query a JSON search endpoint with strict bounds and safe failure behavior."""

    def __init__(self, endpoint: str = "https://api.duckduckgo.com/", timeout: float = 4.0, max_results: int = 5) -> None:
        if timeout <= 0 or max_results <= 0:
            raise ValueError("timeout and max_results must be positive")
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_results = max_results

    def search(self, query: str) -> list[WebSearchResult]:
        """Return bounded results; external failures become an empty result set."""

        if not query.strip():
            raise ValueError("query must not be empty")
        url = f"{self.endpoint}?q={quote(query.strip())}&format=json&no_html=1&skip_disambig=1"
        request = Request(url, headers={"User-Agent": "Atlas-Knowledge-Assistant/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        results: list[WebSearchResult] = []
        for item in payload.get("RelatedTopics", []):
            if "Text" in item and "FirstURL" in item:
                results.append(WebSearchResult(item["Text"], item["FirstURL"], item["Text"]))
            if len(results) >= self.max_results:
                break
        return results
