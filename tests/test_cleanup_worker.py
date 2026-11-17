import importlib

from agentic_rag.config import Settings

app_module = importlib.import_module("agentic_rag.api.app")


class FakeStorage:
    def __init__(self, _root: str) -> None:
        pass

    def purge_recycle_bin(self, _retention_days: int) -> list[str]:
        return ["old.txt", "expired.pdf"]


class FakeIngestion:
    def __init__(self) -> None:
        self.events: list[tuple[str, str, str]] = []

    def record(self, action: str, name: str, detail: str) -> None:
        self.events.append((action, name, detail))


def test_cleanup_worker_records_successful_purge(monkeypatch) -> None:
    ingestion = FakeIngestion()
    monkeypatch.setattr(app_module, "DocumentStorageService", FakeStorage)
    monkeypatch.setattr(app_module, "ingestion_repository", ingestion)
    status: dict[str, object] = {"last_started_at": None, "last_run_at": None, "last_removed_count": 0, "last_error": None}

    removed = app_module.recycle_bin_cleanup_once(Settings(_env_file=None), status)

    assert removed == ["old.txt", "expired.pdf"]
    assert status["last_removed_count"] == 2
    assert status["last_run_at"] is not None
    assert status["last_error"] is None
    assert len(ingestion.events) == 2


def test_cleanup_worker_records_metrics_and_history(monkeypatch, tmp_path) -> None:
    ingestion = FakeIngestion()
    monkeypatch.setattr(app_module, "DocumentStorageService", FakeStorage)
    monkeypatch.setattr(app_module, "ingestion_repository", ingestion)
    repository = app_module.SQLiteOperationalRepository(tmp_path / "operations.db")
    status: dict[str, object] = {"last_started_at": None, "last_run_at": None, "last_removed_count": 0, "last_error": None}

    app_module.recycle_bin_cleanup_once(Settings(_env_file=None), status, repository)

    assert repository.recent_cleanup(1)[0]["status"] == "success"
    assert repository.recent_cleanup(1)[0]["removed_count"] == 2
    assert repository.recent_cleanup(1)[0]["removed_names"] == ["old.txt", "expired.pdf"]
    assert repository.recent_cleanup(1)[0]["duration_ms"] >= 0
    assert repository.cleanup_summary()["total_removed"] == 2
    assert repository.cleanup_summary()["success_rate"] == 1.0


def test_cleanup_worker_reports_storage_errors(monkeypatch) -> None:
    class BrokenStorage(FakeStorage):
        def purge_recycle_bin(self, _retention_days: int) -> list[str]:
            raise OSError("recycle directory unavailable")

    monkeypatch.setattr(app_module, "DocumentStorageService", BrokenStorage)
    status: dict[str, object] = {"last_started_at": None, "last_run_at": None, "last_removed_count": 0, "last_error": None}

    assert app_module.recycle_bin_cleanup_once(Settings(_env_file=None), status) == []
    assert status["last_error"] == "recycle directory unavailable"
    assert status["last_run_at"] is None


