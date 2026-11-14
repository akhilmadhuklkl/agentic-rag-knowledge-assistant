from agentic_rag.embeddings.local import LocalHashEmbeddingProvider


def test_local_embeddings_are_deterministic_and_normalized() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=32)

    first = provider.embed("refund policy")
    second = provider.embed("refund policy")

    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9


def test_empty_text_returns_zero_vector() -> None:
    assert LocalHashEmbeddingProvider(dimensions=8).embed("") == [0.0] * 8
