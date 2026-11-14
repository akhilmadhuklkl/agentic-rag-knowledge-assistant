from agentic_rag.conversations.store import ConversationStore


def test_conversation_store_creates_and_retrieves_turns() -> None:
    store = ConversationStore(max_turns=2)
    conversation_id = store.create()
    store.append(conversation_id, "What is the refund policy?", "Refunds are available.")

    history = store.get(conversation_id)

    assert len(conversation_id) == 32
    assert history[0].question == "What is the refund policy?"


def test_conversation_store_limits_history_and_builds_context() -> None:
    store = ConversationStore(max_turns=2)
    conversation_id = store.create()
    store.append(conversation_id, "first", "one")
    store.append(conversation_id, "second", "two")
    store.append(conversation_id, "third", "three")

    history = store.get(conversation_id)
    contextual = store.contextual_query("follow up", history)

    assert [turn.question for turn in history] == ["second", "third"]
    assert "Previous conversation topics" in contextual
    assert "Current question: follow up" in contextual
