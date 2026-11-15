"""Domain models produced by document ingestion."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable piece of a source document with citation metadata."""

    chunk_id: str
    text: str
    source_path: str
    source_name: str
    file_type: str
    page_number: int | None
    chunk_index: int
    authority: str = "standard"
    freshness: str = "unspecified"


@dataclass(frozen=True)
class SourcePage:
    """Extracted source text before it is split into chunks."""

    source_path: Path
    source_name: str
    file_type: str
    page_number: int | None
    text: str
