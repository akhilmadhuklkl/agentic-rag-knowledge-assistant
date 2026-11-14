from pathlib import Path

import pytest

from agentic_rag.storage.documents import DocumentStorageService

RUNTIME = Path(__file__).parent / ".runtime" / "documents"


def test_document_storage_sanitizes_names_and_lists_documents() -> None:
    service = DocumentStorageService(RUNTIME)

    saved = service.save("../refund policy.md", b"refunds within thirty days")

    assert saved.name == "refund_policy.md"
    assert saved.size_bytes > 0
    assert [document.name for document in service.list_documents()] == ["refund_policy.md"]


def test_document_storage_rejects_unsupported_files() -> None:
    service = DocumentStorageService(RUNTIME)

    with pytest.raises(ValueError, match="Unsupported file type"):
        service.save("malware.exe", b"not a document")
