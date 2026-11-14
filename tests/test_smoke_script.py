from pathlib import Path


def test_smoke_script_exists_and_targets_live_api() -> None:
    script = (Path(__file__).parents[1] / "scripts" / "smoke_test.py").read_text()

    assert "http://127.0.0.1:8000" in script
    assert "/knowledge/ask" in script
    assert "Local API smoke test passed" in script
