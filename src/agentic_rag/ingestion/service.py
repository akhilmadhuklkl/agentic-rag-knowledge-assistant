"""Load local documents and turn them into citation-aware chunks."""

import re
from pathlib import Path

from pypdf import PdfReader

from agentic_rag.ingestion.models import DocumentChunk, SourcePage

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


class DocumentIngestionService:
    """Extract, clean, and chunk supported local document files."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: str | Path) -> list[DocumentChunk]:
        """Extract one supported file and return citation-ready chunks."""

        source_path = Path(path)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type {source_path.suffix!r}; supported: {supported}")

        return self._chunk_pages(self._extract_pages(source_path))

    def ingest_directory(self, directory: str | Path) -> list[DocumentChunk]:
        """Ingest supported files directly inside a directory in stable order."""

        root = Path(directory)
        if not root.is_dir():
            raise NotADirectoryError(root)
        chunks: list[DocumentChunk] = []
        for path in sorted(root.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                chunks.extend(self.ingest_file(path))
        return chunks

    def _extract_pages(self, path: Path) -> list[SourcePage]:
        file_type = path.suffix.lower().lstrip(".")
        if file_type == "pdf":
            reader = PdfReader(str(path))
            return [
                SourcePage(path, path.name, file_type, number, page.extract_text() or "")
                for number, page in enumerate(reader.pages, start=1)
            ]

        text = path.read_text(encoding="utf-8")
        return [SourcePage(path, path.name, file_type, None, text)]

    def _chunk_pages(self, pages: list[SourcePage]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for page in pages:
            text = self._clean_text(page.text)
            if not text:
                continue
            authority, freshness = self._metadata(text)
            for index, chunk_text in enumerate(self._split_text(text)):
                page_marker = page.page_number or 1
                chunk_id = f"{page.source_name}:{page_marker}:{index}"
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        source_path=str(page.source_path),
                        source_name=page.source_name,
                        file_type=page.file_type,
                        page_number=page.page_number,
                        chunk_index=index,
                        authority=authority,
                        freshness=freshness,
                    )
                )
        return chunks

    @staticmethod
    def _metadata(text: str) -> tuple[str, str]:
        """Read optional structured authority metadata from document headers."""

        header = "\n".join(text.splitlines()[:8]).lower()
        authority = "authoritative" if "authority: authoritative" in header else "standard"
        if "authority: legacy" in header:
            authority = "legacy"
        freshness = "current" if "status: current" in header else "unspecified"
        if "status: historical" in header:
            freshness = "historical"
        return authority, freshness

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start + self.chunk_size // 2:
                    end = boundary
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return [chunk for chunk in chunks if chunk]
