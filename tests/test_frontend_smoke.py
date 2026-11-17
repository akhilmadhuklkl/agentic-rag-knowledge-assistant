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
    assert "Updating…" in JAVASCRIPT
    assert "bulk-metadata-undo" in JAVASCRIPT
    assert "atlas-metadata-filters" in JAVASCRIPT
    assert "clear-metadata-selection" in JAVASCRIPT
    assert "atlas-metadata-selection" in JAVASCRIPT
    assert "select-visible-metadata" in JAVASCRIPT
    assert "Select all matching" in JAVASCRIPT
    assert "aria-label" in JAVASCRIPT
    assert "external-source-link" in JAVASCRIPT
    assert "source-domain" in JAVASCRIPT
    assert "web-trust-label" in JAVASCRIPT
    assert "web-quality" in JAVASCRIPT
    assert "web-search-toggle" in JAVASCRIPT
    assert "Web results come from external sources" in JAVASCRIPT
    assert "savedSelection.clear" in JAVASCRIPT
    assert "select-none-metadata" in JAVASCRIPT
    assert "metadata-selection-count" in JAVASCRIPT
    assert "/knowledge/documents/bulk-delete" in JAVASCRIPT
    assert "Delete selected" in JAVASCRIPT
    assert "danger-button" in JAVASCRIPT
    assert 'id="recycle-list"' in HTML
    assert "/knowledge/documents/recycle-bin" in JAVASCRIPT
    assert "Restore & re-index" in JAVASCRIPT
    assert "/knowledge/documents/recycle-bin/purge" in JAVASCRIPT
    assert "Permanently clean up" in JAVASCRIPT
    assert "age_days" in JAVASCRIPT
    assert "recycle_bin_retention_days" in JAVASCRIPT
    assert "/health/recycle-bin" in JAVASCRIPT
    assert "last_removed_count" in JAVASCRIPT
    assert "recycle-history" in JAVASCRIPT
    assert "/health/recycle-bin/run" in JAVASCRIPT
    assert "Run cleanup now" in JAVASCRIPT
    assert "View cleanup report" in JAVASCRIPT
    assert "removed_names" in JAVASCRIPT
    assert "/health/recycle-bin/export" in JAVASCRIPT
    assert "data-cleanup-export" in JAVASCRIPT
    assert "cleanup-from" in JAVASCRIPT
    assert "cleanup-status" in JAVASCRIPT
    assert "from_date" in JAVASCRIPT
    assert "/health/recycle-bin/history" in JAVASCRIPT
    assert "Load older cleanup runs" in JAVASCRIPT


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
    assert "/health/readiness" in JAVASCRIPT
    assert "/health/history" in JAVASCRIPT
    assert "/health/diagnostics" in JAVASCRIPT
    assert 'id="export-diagnostics"' in HTML
    assert 'id="quarantine-activity"' in HTML
    assert "/knowledge/security/quarantine" in JAVASCRIPT
    assert "/knowledge/security/quarantine/files" in JAVASCRIPT
    assert "deleteQuarantinedFile" in JAVASCRIPT
    assert "quarantine-delete" in JAVASCRIPT
    assert 'id="quarantine-load-more"' in HTML
    assert "quarantine-load-more" in JAVASCRIPT
    assert "quarantine_deleted" in JAVASCRIPT
    assert "errors-only" in JAVASCRIPT


def test_backup_application_is_preserved() -> None:
    backup = Path(__file__).parents[1] / "streamlit_backup"
    assert backup.is_dir()
    assert any(backup.iterdir())
