"""Small in-memory metrics registry for local observability."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class MetricsRegistry:
    """Thread-safe counters and duration samples for one process."""

    def __init__(self) -> None:
        self._counters: Counter[str] = Counter()
        self._durations: dict[str, list[float]] = {}
        self._lock = Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._durations.setdefault(name, []).append(value)

    @contextmanager
    def timing(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            self.observe(name, (perf_counter() - started) * 1000)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            durations = {
                name: {
                    "count": len(values),
                    "total_ms": round(sum(values), 2),
                    "average_ms": round(sum(values) / len(values), 2),
                }
                for name, values in self._durations.items()
                if values
            }
            return {"counters": dict(self._counters), "durations": durations}


metrics = MetricsRegistry()
