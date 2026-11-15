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
    styles = (root / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert 'id="file-upload"' in html
    assert 'id="load-sources"' in html
    assert 'id="new-conversation"' in html
    assert "/knowledge/documents/upload" in javascript
    assert "conversation_id" in javascript
    assert "/knowledge/conversations" in javascript
    assert "MINIMUM_RESPONSE_LOADER_MS" in javascript
    assert "waitForMinimumLoaderTime" in javascript
    assert "animateAnswer" in javascript
    assert "requestAnimationFrame" in javascript
    assert 'event.key==="Enter"' in javascript
    assert 'requestSubmit()' in javascript
    assert "escapeHtml" in javascript
    assert "source-excerpt" in javascript
    assert "deleteDocument" in javascript
    assert "/knowledge/documents/" in javascript
    assert "reindexDocument" in javascript
    assert "/knowledge/index/file" in javascript
    assert 'id="source-preview"' in html
    assert "openSourcePreview" in javascript
    assert "View source excerpt" in javascript
    assert "/metrics" in javascript
    assert "verified-rate" in html
    assert 'data-rating="helpful"' in html
    assert "/knowledge/feedback" in javascript
    assert "/knowledge/feedback/summary" in javascript
    assert "feedback-rate" in html
    assert "/knowledge/evaluation/run" in javascript or "/knowledge/evaluation/run" in html
    assert ".answer-card{display:block}" in styles
    assert ".answer-loader.hidden,.answer-content.hidden{display:none}" in styles
    assert ".answer-text{max-height:360px;overflow-y:auto" in styles
    assert "Activity" not in html
