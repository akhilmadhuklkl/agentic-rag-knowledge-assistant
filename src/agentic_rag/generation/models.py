"""Interfaces for language-model-backed answer generation."""

from typing import Protocol


class AnswerModel(Protocol):
    """Minimal interface required by the grounded answer service."""

    def generate(self, prompt: str) -> str:
        """Generate text from a grounded prompt."""
