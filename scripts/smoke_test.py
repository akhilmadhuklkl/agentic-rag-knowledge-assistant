"""Smoke test the running local FastAPI service.

Usage:
    python scripts/smoke_test.py
"""

from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
FIXTURE = Path(__file__).parents[1] / "tests" / "fixtures" / "policy.md"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        health = client.get("/health")
        health.raise_for_status()
        assert health.json()["status"] == "ok"

        indexed = client.post("/knowledge/index/file", json={"path": str(FIXTURE)})
        indexed.raise_for_status()
        assert indexed.json()["chunk_count"] >= 1

        answer = client.post(
            "/knowledge/ask",
            json={"query": "What is the refund policy?", "top_k": 1},
        )
        answer.raise_for_status()
        payload = answer.json()
        assert payload["grounded"] is True
        assert payload["verified"] is True
        assert payload["citations"]
        print("Local API smoke test passed")


if __name__ == "__main__":
    main()
