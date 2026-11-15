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
            connection.execute("CREATE TABLE IF NOT EXISTS index_manifest (source_path TEXT PRIMARY KEY, fingerprint TEXT NOT NULL)")

    def upsert_document(self, document: StoredDocument) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO documents (name, path, size_bytes, file_type) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET path=excluded.path, size_bytes=excluded.size_bytes, file_type=excluded.file_type",
                (document.name, document.path, document.size_bytes, document.file_type),
            )

    def list_documents(self) -> list[StoredDocument]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name, path, size_bytes, file_type FROM documents ORDER BY name").fetchall()
        return [StoredDocument(*row) for row in rows]

    def get_fingerprint(self, source_path: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute("SELECT fingerprint FROM index_manifest WHERE source_path = ?", (source_path,)).fetchone()
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

    def record(self, dataset: str, report: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO evaluation_runs (dataset, total_cases, retrieval_hit_rate, answer_match_rate, citation_accuracy, refusal_accuracy) VALUES (?, ?, ?, ?, ?, ?)",
                (dataset, report["total_cases"], report["retrieval_hit_rate"], report["answer_match_rate"], report["citation_accuracy"], report["refusal_accuracy"]),
            )

    def recent(self, dataset: str | None = None, limit: int = 10) -> list[dict[str, object]]:
        query = "SELECT dataset, total_cases, retrieval_hit_rate, answer_match_rate, citation_accuracy, refusal_accuracy, created_at FROM evaluation_runs"
        values: tuple[object, ...] = ()
        if dataset:
            query += " WHERE dataset = ?"
            values = (dataset,)
        query += " ORDER BY id DESC LIMIT ?"
        values += (limit,)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        keys = ("dataset", "total_cases", "retrieval_hit_rate", "answer_match_rate", "citation_accuracy", "refusal_accuracy", "created_at")
        return [dict(zip(keys, row, strict=True)) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
