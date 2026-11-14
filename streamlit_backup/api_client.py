"""Small HTTP client used by the Streamlit backup interface."""

import httpx


class AssistantApiClient:
    """Call the local assistant API without owning domain logic."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def index_directory(self, path: str) -> dict:
        response = httpx.post(
            f"{self.base_url}/knowledge/index/directory",
            json={"path": path},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def upload_document(self, filename: str, content: bytes) -> dict:
        response = httpx.post(
            f"{self.base_url}/knowledge/documents/upload",
            files={"file": (filename, content)},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def list_documents(self) -> dict:
        response = httpx.get(f"{self.base_url}/knowledge/documents", timeout=30)
        response.raise_for_status()
        return response.json()

    def ask(self, question: str, top_k: int = 5) -> dict:
        response = httpx.post(
            f"{self.base_url}/knowledge/ask",
            json={"query": question, "top_k": top_k},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
