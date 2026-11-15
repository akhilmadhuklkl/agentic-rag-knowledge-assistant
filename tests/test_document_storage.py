from pathlib import Path

import pytest

from agentic_rag.storage.documents import DocumentStorageService

RUNTIME = Path(__file__).parent / ".runtime" / "documents"


def test_document_storage_sanitizes_names_and_lists_documents(tmp_path: Path) -> None:
    service = DocumentStorageService(tmp_path)

    saved = service.save("../refund policy.md", b"refunds within thirty days")

    assert saved.name == "refund_policy.md"
    assert saved.size_bytes > 0
    assert [document.name for document in service.list_documents()] == ["refund_policy.md"]


def test_document_storage_rejects_unsupported_files(tmp_path: Path) -> None:
    service = DocumentStorageService(tmp_path)

    with pytest.raises(ValueError, match="Unsupported file type"):
        service.save("malware.exe", b"not a document")


def test_document_storage_rejects_duplicate_names_and_content(tmp_path: Path) -> None:
    service = DocumentStorageService(tmp_path)
    service.save("policy.md", b"refund policy")

    with pytest.raises(ValueError, match="already exists"):
        service.save("policy.md", b"different content")
    with pytest.raises(ValueError, match="duplicates"):
        service.save("copy.txt", b"refund policy")
