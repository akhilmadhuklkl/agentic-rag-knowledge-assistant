from agentic_rag.api.routes import (
    conversation_store,
    delete_conversation,
    get_conversation,
    list_conversations,
)


def test_conversation_management_routes_work() -> None:
    conversation_id = conversation_store.create()
    conversation_store.append(conversation_id, "hello", "world")

    assert conversation_id in list_conversations().conversation_ids
    assert get_conversation(conversation_id).turns[0].answer == "world"
    assert delete_conversation(conversation_id)["status"] == "deleted"
