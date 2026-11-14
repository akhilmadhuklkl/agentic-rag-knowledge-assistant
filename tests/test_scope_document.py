from pathlib import Path


def test_scope_document_mentions_mvp_boundaries() -> None:
    scope = (Path(__file__).parents[1] / "docs" / "project-scope.md").read_text()
    assert "MVP capabilities" in scope
    assert "Deferred" in scope
    assert "Definition of done" in scope
