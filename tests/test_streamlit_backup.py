from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_streamlit_backup_is_a_thin_api_client() -> None:
    app_text = (ROOT / "streamlit_backup" / "app.py").read_text(encoding="utf-8")
    client_text = (ROOT / "streamlit_backup" / "api_client.py").read_text(encoding="utf-8")

    assert "AssistantApiClient" in app_text
    assert "/knowledge/ask" in client_text
    assert "DocumentIndexingService" not in app_text
