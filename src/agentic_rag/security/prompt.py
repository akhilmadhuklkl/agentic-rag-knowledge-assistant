"""Guard boundaries between trusted instructions and untrusted evidence."""

import re


def validate_question(question: str, max_length: int = 2000) -> str:
    """Normalize and validate user questions before tool execution."""

    normalized = question.strip()
    if not normalized:
        raise ValueError("question must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"question exceeds the {max_length}-character limit")
    return normalized


def evidence_block(text: str) -> str:
    """Fence retrieved text so it cannot be confused with system instructions."""

    cleaned = re.sub(r"\x00", " ", text).strip()
    return f"<retrieved_evidence>\n{cleaned}\n</retrieved_evidence>"
