from pathlib import Path

from agentic_rag.api import routes
from agentic_rag.api.schemas import BulkDocumentMetadataUpdate, DocumentMetadataUpdate
from agentic_rag.storage.documents import StoredDocument


class FakeRepository:
    def __init__(self, document: StoredDocument, *additional: StoredDocument) -> None:
        self.documents = [document, *additional]
        self.updated: tuple[str, str, str] | None = None

    def list_documents(self) -> list[StoredDocument]:
        return self.documents

    def update_metadata(self, name: str, authority: str, freshness: str) -> None:
        self.updated = (name, authority, freshness)


class FakeVectorStore:
    def __init__(self) -> None:
        self.updated: tuple[str, str, str] | None = None

    def update_source_metadata(self, path: str, authority: str, freshness: str) -> None:
        self.updated = (path, authority, freshness)


class FakeIndexing:
    def __init__(self) -> None:
        self.vector_store = FakeVectorStore()
        self.reindexed: str | None = None

    def reindex_file(self, path: str) -> None:
        self.reindexed = path


class FakeActivity:
    def __init__(self) -> None:
        self.event: tuple[str, str, str] | None = None

    def record(self, action: str, name: str, detail: str) -> None:
        self.event = (action, name, detail)


def test_metadata_route_persists_and_reindexes_document(monkeypatch, tmp_path: Path) -> None:
    path = str(tmp_path / "policy.md")
    document = StoredDocument("policy.md", path, 10, "md")
    repository = FakeRepository(document)
    indexing = FakeIndexing()
    activity = FakeActivity()
    monkeypatch.setattr(routes, "document_repository", repository)
    monkeypatch.setattr(routes, "ingestion_repository", activity)

    response = routes.update_document_metadata(
        "policy.md",
        DocumentMetadataUpdate(authority="authoritative", freshness="current"),
        storage=object(),
        indexing=indexing,
    )

    assert response.authority == "authoritative"
    assert response.freshness == "current"
    assert repository.updated == ("policy.md", "authoritative", "current")
    assert indexing.reindexed == path
    assert indexing.vector_store.updated == (path, "authoritative", "current")
    assert activity.event and activity.event[0] == "metadata_updated"


def test_bulk_metadata_route_updates_and_reindexes_all_documents(monkeypatch, tmp_path: Path) -> None:
    first = StoredDocument("one.md", str(tmp_path / "one.md"), 10, "md")
    second = StoredDocument("two.md", str(tmp_path / "two.md"), 20, "md")
    repository = FakeRepository(first, second)
    indexing = FakeIndexing()
    activity = FakeActivity()
    monkeypatch.setattr(routes, "document_repository", repository)
    monkeypatch.setattr(routes, "ingestion_repository", activity)

    response = routes.update_documents_metadata(
        BulkDocumentMetadataUpdate(names=["one.md", "two.md"], authority="legacy", freshness="historical"),
        indexing=indexing,
    )

    assert [document.authority for document in response.updated] == ["legacy", "legacy"]
    assert [document.freshness for document in response.updated] == ["historical", "historical"]
    assert response.errors == []
    assert repository.updated == ("two.md", "legacy", "historical")
    assert indexing.reindexed == str(tmp_path / "two.md")
    assert indexing.vector_store.updated == (str(tmp_path / "two.md"), "legacy", "historical")


def test_bulk_metadata_route_rejects_unknown_document(monkeypatch, tmp_path: Path) -> None:
    repository = FakeRepository(StoredDocument("one.md", str(tmp_path / "one.md"), 10, "md"))
    monkeypatch.setattr(routes, "document_repository", repository)

    response = routes.update_documents_metadata(
        BulkDocumentMetadataUpdate(names=["missing.md"], authority="standard", freshness="unspecified"),
        indexing=FakeIndexing(),
    )

    assert response.updated == []
    assert response.errors == ["missing.md: document not found"]
