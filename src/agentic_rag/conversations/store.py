"""Local conversation history separated from knowledge-base storage."""

from dataclasses import dataclass
from threading import Lock
from uuid import uuid4


@dataclass(frozen=True)
class ConversationTurn:
    """One user/assistant exchange."""

    question: str
    answer: str


class ConversationStore:
    """Thread-safe in-memory conversation store for the local MVP."""

    def __init__(self, max_turns: int = 20) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns
        self._conversations: dict[str, list[ConversationTurn]] = {}
        self._lock = Lock()

    def create(self) -> str:
        conversation_id = uuid4().hex
        with self._lock:
            self._conversations[conversation_id] = []
        return conversation_id

    def get(self, conversation_id: str) -> list[ConversationTurn]:
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, question: str, answer: str) -> None:
        with self._lock:
            turns = self._conversations.setdefault(conversation_id, [])
            turns.append(ConversationTurn(question, answer))
            del turns[:-self.max_turns]

    @staticmethod
    def contextual_query(question: str, history: list[ConversationTurn]) -> str:
        """Add recent user topics to a follow-up query without hiding the current question."""

        if not history:
            return question
        previous_questions = " ".join(turn.question for turn in history[-3:])
        return f"Previous conversation topics: {previous_questions}\nCurrent question: {question}"
