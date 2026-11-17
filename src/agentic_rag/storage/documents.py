"""Safe local storage for knowledge-base documents."""

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_rag.ingestion.service import SUPPORTED_EXTENSIONS
from agentic_rag.security.file_scan import BasicDocumentScanner


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

    def __init__(self, root: str | Path = "data/documents", max_size_bytes: int = 10_000_000, repository: Any | None = None, scanner: Any | None = None, quarantine_root: str | Path | None = None, quarantine_audit: Any | None = None, recycle_root: str | Path | None = None) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be positive")
        self.root = Path(root)
        self.max_size_bytes = max_size_bytes
        self.repository = repository
        self.scanner = scanner or BasicDocumentScanner()
        self.quarantine_root = Path(quarantine_root) if quarantine_root else self.root.parent / "quarantine"
        self.quarantine_audit = quarantine_audit
        self.recycle_root = Path(recycle_root) if recycle_root else self.root.parent / "recycle_bin"
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
        scan = self.scanner.scan(safe_name, content)
        if not scan.safe:
            self._quarantine(safe_name, content)
            if self.quarantine_audit:
                self.quarantine_audit.record("quarantined", safe_name, scan.reason)
            raise ValueError(f"Upload quarantined: {scan.reason}")

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

    def _quarantine(self, filename: str, content: bytes) -> Path:
        """Store a rejected upload outside the indexed document root."""

        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = self.quarantine_root / f"{hashlib.sha256(content).hexdigest()}-{Path(filename).name}"
        destination.write_bytes(content)
        return destination

    def list_quarantine(self) -> list[str]:
        """List quarantined file names without exposing their contents."""

        if not self.quarantine_root.is_dir():
            return []
        return sorted(path.name for path in self.quarantine_root.iterdir() if path.is_file())

    def delete_quarantined(self, name: str) -> bool:
        """Delete one quarantined file after validating it stays in quarantine."""

        candidate = (self.quarantine_root / Path(name).name).resolve()
        if candidate.parent != self.quarantine_root.resolve() or not candidate.is_file():
            return False
        candidate.unlink()
        return True

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
        """Move one stored document to the recycle bin and return its metadata."""

        path = self.root / self._safe_filename(name)
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(name)
        document = self._metadata(path)
        self.recycle_root.mkdir(parents=True, exist_ok=True)
        destination = self.recycle_root / path.name
        if destination.exists():
            destination.unlink()
        path.replace(destination)
        if self.repository:
            self.repository.delete_document(document.name)
            self.repository.delete_fingerprint(document.path)
        return document

    def list_recycle_bin(self) -> list[str]:
        """List documents available for restoration."""

        if not self.recycle_root.is_dir():
            return []
        return sorted(path.name for path in self.recycle_root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)

    def recycle_bin_details(self) -> list[dict[str, object]]:
        """Return recycle-bin names and retention metadata."""

        details = []
        for name in self.list_recycle_bin():
            path = self.recycle_root / name
            deleted_at = datetime.fromtimestamp(path.stat().st_mtime, UTC)
            details.append({"name": name, "deleted_at": deleted_at.isoformat(), "age_days": max(0, (datetime.now(UTC) - deleted_at).days)})
        return details

    def permanently_delete_recycled(self, name: str) -> bool:
        """Permanently remove one recycled document after explicit approval."""

        candidate = (self.recycle_root / self._safe_filename(name)).resolve()
        if candidate.parent != self.recycle_root.resolve() or not candidate.is_file():
            return False
        candidate.unlink()
        return True

    def purge_recycle_bin(self, max_age_days: int) -> list[str]:
        """Permanently remove recycled documents at or beyond the age limit."""

        if max_age_days < 1:
            raise ValueError("max_age_days must be positive")
        removed = []
        for item in self.recycle_bin_details():
            if int(item["age_days"]) >= max_age_days and self.permanently_delete_recycled(str(item["name"])):
                removed.append(str(item["name"]))
        return removed

    def restore(self, name: str) -> StoredDocument:
        """Restore one recycled document back into active storage."""

        safe_name = self._safe_filename(name)
        source = self.recycle_root / safe_name
        destination = self.root / safe_name
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise FileNotFoundError(name)
        if destination.exists():
            raise FileExistsError(f"A document named '{safe_name}' already exists")
        source.replace(destination)
        document = self._metadata(destination)
        if self.repository:
            self.repository.upsert_document(document)
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
