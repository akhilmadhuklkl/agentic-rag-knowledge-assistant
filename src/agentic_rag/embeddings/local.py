"""Deterministic, dependency-light local embeddings for development."""

import hashlib
import math
import re


class LocalHashEmbeddingProvider:
    """Map tokens into a fixed-size vector using signed feature hashing.

    This is intentionally deterministic and requires no model download. It is a
    development baseline, not a replacement for a trained embedding model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    @property
    def provider_id(self) -> str:
        """Stable identifier stored with vector indexes."""

        return f"local-hash-v1:{self.dimensions}"

    def embed(self, text: str) -> list[float]:
        """Create a normalized vector for one text value."""

        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[\w'-]+", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return self._normalize(vector)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Create vectors in input order."""

        return [self.embed(text) for text in texts]

    @staticmethod
    def _normalize(vector: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]
