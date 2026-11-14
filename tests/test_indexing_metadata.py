from pathlib import Path

from agentic_rag.indexing.service import DocumentIndexingService
from agentic_rag.storage.repository import SQLiteDocumentRepository


def test_indexing_registers_source_in_metadata_repository(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("The product supports document search.", encoding="utf-8")
    repository = SQLiteDocumentRepository(tmp_path / "documents.db")
    indexing = DocumentIndexingService(
        tmp_path / "vectors.json",
        metadata_repository=repository,
    )

    indexing.index_file(source)

    documents = repository.list_documents()
    assert len(documents) == 1
    assert documents[0].name == "knowledge.md"
    assert documents[0].path == str(source)
