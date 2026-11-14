from agentic_rag.metrics import MetricsRegistry


def test_metrics_registry_tracks_counters_and_durations() -> None:
    registry = MetricsRegistry()
    registry.increment("requests_total", 2)
    registry.observe("request_duration_ms", 10.0)
    registry.observe("request_duration_ms", 20.0)

    snapshot = registry.snapshot()

    assert snapshot["counters"] == {"requests_total": 2}
    assert snapshot["durations"]["request_duration_ms"] == {
        "count": 2,
        "total_ms": 30.0,
        "average_ms": 15.0,
    }
