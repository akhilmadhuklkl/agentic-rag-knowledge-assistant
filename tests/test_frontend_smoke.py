"""Smoke tests for the product UI's critical static hooks."""

from pathlib import Path

FRONTEND = Path(__file__).parents[1] / "frontend"
HTML = (FRONTEND / "index.html").read_text(encoding="utf-8")
JAVASCRIPT = (FRONTEND / "app.js").read_text(encoding="utf-8")


def test_primary_question_workflow_hooks_are_present() -> None:
    assert 'id="ask-form"' in HTML
    assert 'id="question"' in HTML
    assert 'id="answer-loader"' in HTML
    assert 'id="answer-content"' in HTML
    assert 'requestSubmit()' in JAVASCRIPT
    assert "setAnswerLoading(true)" in JAVASCRIPT


def test_document_management_hooks_are_present() -> None:
    assert 'id="file-upload"' in HTML
    assert "/knowledge/documents/upload" in JAVASCRIPT
    assert "reindexStale" in JAVASCRIPT
    assert "deleteDocument" in JAVASCRIPT
    assert "document will be re-indexed" in JAVASCRIPT
    assert "metadata-history-dialog" in JAVASCRIPT


def test_quality_control_hooks_are_present() -> None:
    assert 'id="run-evaluation"' in HTML
    assert 'id="quality-preview"' in HTML
    assert 'id="export-quality-json"' in HTML
    assert 'id="export-quality-csv"' in HTML
    assert "quality_passed" in JAVASCRIPT


def test_health_and_activity_hooks_are_present() -> None:
    assert 'id="knowledge-health"' in HTML
    assert 'id="ingestion-activity"' in HTML
    assert "/health" in JAVASCRIPT
    assert "errors-only" in JAVASCRIPT


def test_backup_application_is_preserved() -> None:
    backup = Path(__file__).parents[1] / "streamlit_backup"
    assert backup.is_dir()
    assert any(backup.iterdir())
