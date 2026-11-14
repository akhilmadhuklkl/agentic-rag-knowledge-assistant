from pathlib import Path

from agentic_rag.conversations.repository import SQLiteConversationRepository
from agentic_rag.conversations.store import ConversationStore


def test_sqlite_repository_persists_conversation_turns() -> None:
    database = Path(__file__).parent / ".runtime" / "assistant.db"
    database.unlink(missing_ok=True)
    repository = SQLiteConversationRepository(database)
    store = ConversationStore(repository=repository)

    conversation_id = store.create()
    store.append(conversation_id, "hello", "world")
    restored = ConversationStore(repository=SQLiteConversationRepository(database))

    assert restored.get(conversation_id)[0].answer == "world"
    assert conversation_id in restored.list_ids()
    assert restored.delete(conversation_id) is True
