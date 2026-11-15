from pathlib import Path

import pytest

from agentic_rag.ingestion.service import DocumentIngestionService

FIXTURES = Path(__file__).parent / "fixtures"


def test_ingests_and_chunks_markdown_with_source_metadata() -> None:
    chunks = DocumentIngestionService().ingest_file(FIXTURES / "policy.md")

    assert len(chunks) == 1
    assert chunks[0].source_name == "policy.md"
    assert chunks[0].file_type == "md"
    assert chunks[0].page_number is None
    assert "Refunds are available" in chunks[0].text


def test_splits_large_text_with_overlap() -> None:
    chunks = DocumentIngestionService(chunk_size=40, chunk_overlap=10).ingest_file(
        FIXTURES / "notes.txt"
    )

    assert len(chunks) > 1
    assert chunks[0].text[-5:] in chunks[1].text


def test_rejects_unsupported_files() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        DocumentIngestionService().ingest_file(FIXTURES / "image.csv")


def test_ingests_supported_files_in_sorted_order() -> None:
    chunks = DocumentIngestionService().ingest_directory(FIXTURES / "directory")

    assert [chunk.source_name for chunk in chunks] == ["a.md", "b.txt"]


def test_reads_authority_and_freshness_metadata() -> None:
    authority, freshness = DocumentIngestionService._metadata(
        "Authority: authoritative\nStatus: current\n\n# Policy\nCurrent rule."
    )

    assert authority == "authoritative"
    assert freshness == "current"
