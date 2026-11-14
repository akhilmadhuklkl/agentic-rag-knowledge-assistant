import pytest

from agentic_rag.reliability import retry_call


def test_retry_call_retries_transient_failures() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    assert retry_call(operation, (TimeoutError,), max_attempts=3) == "ok"
    assert attempts == 3


def test_retry_call_raises_after_bound() -> None:
    with pytest.raises(TimeoutError):
        retry_call(lambda: (_ for _ in ()).throw(TimeoutError("failed")), (TimeoutError,), 2)
