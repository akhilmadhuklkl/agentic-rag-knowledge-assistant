"""SQLite repositories for document metadata and index freshness."""

import sqlite3
from pathlib import Path

from agentic_rag.storage.documents import StoredDocument


class SQLiteDocumentRepository:
    """Persist document metadata and source fingerprints in SQLite."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS documents (name TEXT PRIMARY KEY, path TEXT NOT NULL, size_bytes INTEGER NOT NULL, file_type TEXT NOT NULL)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)").fetchall()}
            if "created_at" not in columns:
                connection.execute("ALTER TABLE documents ADD COLUMN created_at TEXT")
                connection.execute("UPDATE documents SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            for column in ("authority", "freshness"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT NOT NULL DEFAULT 'standard'")
            connection.execute("CREATE TABLE IF NOT EXISTS index_manifest (source_path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)")

    def upsert_document(self, document: StoredDocument) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents (name, path, size_bytes, file_type, authority, freshness) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET path=excluded.path, size_bytes=excluded.size_bytes, file_type=excluded.file_type, authority=excluded.authority, freshness=excluded.freshness",
                (document.name, document.path, document.size_bytes, document.file_type, document.authority, document.freshness),
            )

    def list_documents(self) -> list[StoredDocument]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name, path, size_bytes, file_type, authority, freshness FROM documents ORDER BY name").fetchall()
        return [StoredDocument(*row) for row in rows]

    def update_metadata(self, name: str, authority: str, freshness: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE documents SET authority = ?, freshness = ? WHERE name = ?", (authority, freshness, name))

    def get_fingerprint(self, source_path: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT fingerprint FROM index_manifest WHERE source_path = ?", (source_path,)).fetchone()
        return row[0] if row else None

    def get_created_at(self, name: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT created_at FROM documents WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_fingerprint(self, source_path: str, fingerprint: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO index_manifest (source_path, fingerprint) VALUES (?, ?) "
                "ON CONFLICT(source_path) DO UPDATE SET fingerprint=excluded.fingerprint",
                (source_path, fingerprint),
            )

    def delete_document(self, name: str) -> bool:
        with self._connect() as connection:
            result = connection.execute("DELETE FROM documents WHERE name = ?", (name,))
        return result.rowcount > 0

    def delete_fingerprint(self, source_path: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM index_manifest WHERE source_path = ?", (source_path,))

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class SQLiteFeedbackRepository:
    """Persist answer-quality feedback in SQLite."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS answer_feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, rating TEXT NOT NULL, conversation_id TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")

    def record(self, rating: str, conversation_id: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO answer_feedback (rating, conversation_id) VALUES (?, ?)",
                (rating, conversation_id),
            )

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT rating, COUNT(*) FROM answer_feedback GROUP BY rating"
            ).fetchall()
        values = {str(rating): int(count) for rating, count in rows}
        return {
            "total": sum(values.values()),
            "helpful": values.get("helpful", 0),
            "needs_improvement": values.get("needs_improvement", 0),
        }

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class SQLiteEvaluationRepository:
    """Persist compact benchmark results for trend comparisons."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS evaluation_runs (id INTEGER PRIMARY KEY AUTOINCREMENT, dataset TEXT NOT NULL, total_cases INTEGER NOT NULL, retrieval_hit_rate REAL NOT NULL, answer_match_rate REAL NOT NULL, citation_accuracy REAL NOT NULL, refusal_accuracy REAL NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(evaluation_runs)").fetchall()}
            for column in ("answerable_retrieval_rate", "answerable_citation_accuracy", "refusal_case_accuracy"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE evaluation_runs ADD COLUMN {column} REAL NOT NULL DEFAULT 0")

    def record(self, dataset: str, report: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs (dataset, total_cases, retrieval_hit_rate, answer_match_rate, citation_accuracy, refusal_accuracy, answerable_retrieval_rate, answerable_citation_accuracy, refusal_case_accuracy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (dataset, report["total_cases"], report["retrieval_hit_rate"], report["answer_match_rate"], report["citation_accuracy"], report["refusal_accuracy"], report["answerable_retrieval_rate"], report["answerable_citation_accuracy"], report["refusal_case_accuracy"]),
            )

    def recent(self, dataset: str | None = None, limit: int = 10) -> list[dict[str, object]]:
        query = "SELECT dataset, total_cases, retrieval_hit_rate, answer_match_rate, citation_accuracy, refusal_accuracy, answerable_retrieval_rate, answerable_citation_accuracy, refusal_case_accuracy, created_at FROM evaluation_runs"
        values: tuple[object, ...] = ()
        if dataset:
            query += " WHERE dataset = ?"
            values = (dataset,)
        query += " ORDER BY id DESC LIMIT ?"
        values += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        keys = ("dataset", "total_cases", "retrieval_hit_rate", "answer_match_rate", "citation_accuracy", "refusal_accuracy", "answerable_retrieval_rate", "answerable_citation_accuracy", "refusal_case_accuracy", "created_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class SQLiteIngestionRepository:
    """Persist a compact audit trail of document operations."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS ingestion_activity (id INTEGER PRIMARY KEY AUTOINCREMENT, action TEXT NOT NULL, document_name TEXT NOT NULL, detail TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")

    def record(self, action: str, document_name: str, detail: str) -> None:
        with self._connect() as connection:
            connection.execute("INSERT INTO ingestion_activity (action, document_name, detail) VALUES (?, ?, ?)", (action, document_name, detail))

    def recent(self, limit: int = 20) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT action, document_name, detail, created_at FROM ingestion_activity ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ("action", "document_name", "detail", "created_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class SQLiteLatencyRepository:
    """Persist recent assistant response durations for local trend views."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS latency_history (id INTEGER PRIMARY KEY AUTOINCREMENT, duration_ms REAL NOT NULL, profile TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(latency_history)").fetchall()}
            for column in ("retrieval_ms", "generation_ms", "verification_ms"):
                if column not in columns:
                    connection.execute(f"ALTER TABLE latency_history ADD COLUMN {column} REAL")

    def record(self, duration_ms: float, profile: str, stage_timings: dict[str, float] | None = None) -> None:
        with self._connect() as connection:
            timings = stage_timings or {}
            connection.execute("INSERT INTO latency_history (duration_ms, profile, retrieval_ms, generation_ms, verification_ms) VALUES (?, ?, ?, ?, ?)", (duration_ms, profile, timings.get("retrieval"), timings.get("generation"), timings.get("verification")))

    def recent(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT duration_ms, profile, created_at, retrieval_ms, generation_ms, verification_ms FROM latency_history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ("duration_ms", "profile", "created_at", "retrieval_ms", "generation_ms", "verification_ms")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
