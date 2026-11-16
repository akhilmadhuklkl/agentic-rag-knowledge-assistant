"""Safe local storage for knowledge-base documents."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_rag.ingestion.service import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class StoredDocument:
    """Metadata for a stored source document."""

    name: str
    path: str
    size_bytes: int
    file_type: str
    authority: str = "standard"
    freshness: str = "unspecified"


class DocumentStorageService:
    """Store and list documents under one controlled root directory."""

    def __init__(self, root: str | Path = "data/documents", max_size_bytes: int = 10_000_000, repository: Any | None = None) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        self.root = Path(root)
        self.max_size_bytes = max_size_bytes
        self.repository = repository
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> StoredDocument:
        """Validate and save one uploaded document."""

        safe_name = self._safe_filename(filename)
        if not safe_name:
            raise ValueError("filename must contain a valid name")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or '<none>'}")
        if len(safe_name) > 255:
            raise ValueError("filename is too long")
        if not content:
            raise ValueError("document must not be empty")
        if len(content) > self.max_size_bytes:
            raise ValueError("document exceeds the maximum allowed size")
        if suffix == ".pdf" and not content.startswith(b"%PDF-"):
            raise ValueError("PDF content signature is invalid")
        if suffix in {".md", ".markdown", ".txt"}:
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("text document must be valid UTF-8") from error

        destination = self.root / safe_name
        if destination.exists():
            raise ValueError(f"A document named '{safe_name}' already exists")
        content_hash = hashlib.sha256(content).digest()
        for existing in self.root.iterdir():
            if existing.is_file() and existing.suffix.lower() in SUPPORTED_EXTENSIONS and hashlib.sha256(existing.read_bytes()).digest() == content_hash:
                raise ValueError(f"This file duplicates the existing document '{existing.name}'")
        destination.write_bytes(content)
        document = self._metadata(destination)
        if self.repository:
            self.repository.upsert_document(document)
        return document

    def list_documents(self) -> list[StoredDocument]:
        """Return supported documents in stable filename order."""

        if self.repository:
            return self.repository.list_documents()
        return [
            self._metadata(path)
            for path in sorted(self.root.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    def delete(self, name: str) -> StoredDocument:
        """Delete one stored document and return its metadata."""

        path = self.root / self._safe_filename(name)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(name)
        document = self._metadata(path)
        path.unlink()
        if self.repository:
            self.repository.delete_document(document.name)
            self.repository.delete_fingerprint(document.path)
        return document

    @staticmethod
    def _safe_filename(filename: str) -> str:
        name = Path(filename).name
        return re.sub(r"[^A-Za-z0-9._-]", "_", name)

    @staticmethod
    def _metadata(path: Path) -> StoredDocument:
        return StoredDocument(
            name=path.name,
            path=str(path),
            size_bytes=path.stat().st_size,
            file_type=path.suffix.lower().lstrip("."),
        )
