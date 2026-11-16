"""Upload scanning seam with a conservative local fallback."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScanResult:
    safe: bool
    reason: str = ""


class BasicDocumentScanner:
    """Reject obvious executable payloads before document indexing.

    This is an integration point for ClamAV or another antivirus engine in a
    deployed environment; it is not a replacement for full malware scanning.
    """

    def scan(self, filename: str, content: bytes) -> ScanResult:
        if content.startswith((b"MZ", b"\x7fELF")):
            return ScanResult(False, "executable content signature detected")
        if b"<script" in content.lower() and filename.lower().endswith(".pdf"):
            return ScanResult(False, "script payload detected in PDF")
        return ScanResult(True)
