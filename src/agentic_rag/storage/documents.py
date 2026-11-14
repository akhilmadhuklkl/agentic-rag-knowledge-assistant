"""Safe local storage for knowledge-base documents."""

import re
from dataclasses import dataclass
from pathlib import Path

from agentic_rag.ingestion.service import SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class StoredDocument:
    """Metadata for a stored source document."""

    name: str
    path: str
    size_bytes: int
    file_type: str


class DocumentStorageService:
    """Store and list documents under one controlled root directory."""

    def __init__(self, root: str | Path = "data/documents", max_size_bytes: int = 10_000_000) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        self.root = Path(root)
        self.max_size_bytes = max_size_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: bytes) -> StoredDocument:
        """Validate and save one uploaded document."""

        safe_name = self._safe_filename(filename)
        if not safe_name:
            raise ValueError("filename must contain a valid name")
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {suffix or '<none>'}")
        if len(content) > self.max_size_bytes:
            raise ValueError("document exceeds the maximum allowed size")

        destination = self.root / safe_name
        destination.write_bytes(content)
        return self._metadata(destination)

    def list_documents(self) -> list[StoredDocument]:
        """Return supported documents in stable filename order."""

        return [
            self._metadata(path)
            for path in sorted(self.root.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

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
