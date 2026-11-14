from pathlib import Path

from agentic_rag.api.app import create_app


def test_primary_ui_assets_exist_and_are_mounted() -> None:
    root = Path(__file__).parents[1]
    paths = {route.path for route in create_app().routes}

    assert (root / "frontend" / "index.html").is_file()
    assert (root / "frontend" / "styles.css").is_file()
    assert (root / "frontend" / "app.js").is_file()
    assert "/app" in paths

    html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
    javascript = (root / "frontend" / "app.js").read_text(encoding="utf-8")
    assert 'id="file-upload"' in html
    assert 'id="load-sources"' in html
    assert "/knowledge/documents/upload" in javascript
    assert "Activity" not in html
