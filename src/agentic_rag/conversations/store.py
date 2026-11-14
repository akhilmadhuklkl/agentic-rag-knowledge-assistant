"""Local conversation history separated from knowledge-base storage."""

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant exchange."""

    question: str
    answer: str


class ConversationStore:
    """Thread-safe in-memory conversation store for the local MVP."""

    def __init__(
        self,
        max_turns: int = 20,
        path: str | Path | None = None,
        repository: Any | None = None,
    ) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self.path = Path(path) if path else None
        self.repository = repository
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._lock = Lock()
        self._load()

    def create(self) -> str:
        conversation_id = uuid4().hex
        with self._lock:
            self._conversations[conversation_id] = []
            self._save()
            if self.repository:
                self.repository.create(conversation_id)
        return conversation_id

    def get(self, conversation_id: str) -> list[ConversationTurn]:
        if self.repository:
            return self.repository.get(conversation_id)
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, question: str, answer: str) -> None:
        with self._lock:
            turns = self._conversations.setdefault(conversation_id, [])
            turns.append(ConversationTurn(question, answer))
            del turns[:-self.max_turns]
            self._save()
            if self.repository:
                self.repository.append(conversation_id, turns)

    def list_ids(self) -> list[str]:
        """Return known conversation IDs in stable order."""

        if self.repository:
            return self.repository.list_ids()
        with self._lock:
            return sorted(self._conversations)

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation and report whether it existed."""

        if self.repository:
            return self.repository.delete(conversation_id)
        with self._lock:
            existed = conversation_id in self._conversations
            self._conversations.pop(conversation_id, None)
            if existed:
                self._save()
            return existed

    @staticmethod
    def contextual_query(question: str, history: list[ConversationTurn]) -> str:
        """Add recent user topics to a follow-up query without hiding the current question."""

        if not history:
            return question
        previous_questions = " ".join(turn.question for turn in history[-3:])
        return f"Previous conversation topics: {previous_questions}\nCurrent question: {question}"

    def _load(self) -> None:
        if not self.path or not self.path.is_file():
            return
        values = json.loads(self.path.read_text(encoding="utf-8"))
        self._conversations = {
            conversation_id: [ConversationTurn(**turn) for turn in turns]
            for conversation_id, turns in values.items()
        }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        values = {
            conversation_id: [turn.__dict__ for turn in turns]
            for conversation_id, turns in self._conversations.items()
        }
        self.path.write_text(json.dumps(values, indent=2), encoding="utf-8")
