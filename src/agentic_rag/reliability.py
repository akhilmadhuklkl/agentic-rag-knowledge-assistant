"""Bounded retry helpers for transient service failures."""

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(  # noqa: UP047
    operation: Callable[[], T],
    retry_exceptions: tuple[type[Exception], ...],
    max_attempts: int = 3,
    backoff_seconds: float = 0.0,
) -> T:
    """Retry an operation a bounded number of times."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    for attempt in range(max_attempts):
        try:
            return operation()
        except retry_exceptions:
            if attempt == max_attempts - 1:
                raise
            if backoff_seconds > 0:
                time.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError("retry operation did not return")
