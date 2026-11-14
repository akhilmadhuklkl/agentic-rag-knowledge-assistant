from agentic_rag.observability import (
    create_request_id,
    get_request_id,
    set_request_id,
)


def test_request_ids_are_generated_and_correlated() -> None:
    request_id = create_request_id()
    set_request_id(request_id)

    assert len(request_id) == 32
    assert get_request_id() == request_id
