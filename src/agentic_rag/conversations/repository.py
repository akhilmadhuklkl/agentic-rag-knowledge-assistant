"""Conversation persistence interfaces and a SQLite implementation."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from agentic_rag.conversations.store import ConversationTurn


class SQLiteConversationRepository:
    """Persist conversation turns in a local SQLite database."""

    def __init__(self, path: str | Path = "data/assistant.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversations "
                "(conversation_id TEXT PRIMARY KEY)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS conversation_turns "
                "(conversation_id TEXT NOT NULL, turn_index INTEGER NOT NULL, "
                "question TEXT NOT NULL, answer TEXT NOT NULL, "
                "PRIMARY KEY (conversation_id, turn_index))"
            )

    def create(self, conversation_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id) VALUES (?)",
                (conversation_id,),
            )

    def get(self, conversation_id: str) -> list[ConversationTurn]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT question, answer FROM conversation_turns "
                "WHERE conversation_id = ? ORDER BY turn_index",
                (conversation_id,),
            ).fetchall()
        return [ConversationTurn(question, answer) for question, answer in rows]

    def append(self, conversation_id: str, turns: Iterable[ConversationTurn]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id) VALUES (?)",
                (conversation_id,),
            )
            connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)
            )
            connection.executemany(
                "INSERT INTO conversation_turns "
                "(conversation_id, turn_index, question, answer) VALUES (?, ?, ?, ?)",
                [(conversation_id, index, turn.question, turn.answer) for index, turn in enumerate(turns)],
            )

    def list_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT conversation_id FROM conversations ORDER BY conversation_id"
            ).fetchall()
        return [row[0] for row in rows]

    def delete(self, conversation_id: str) -> bool:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,)
            )
            result = connection.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )
        return result.rowcount > 0

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
