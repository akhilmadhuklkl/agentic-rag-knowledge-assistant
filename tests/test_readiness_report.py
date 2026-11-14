from pathlib import Path


def test_readiness_report_contains_audit_status() -> None:
    report = (Path(__file__).parents[1] / "docs" / "readiness-report.md").read_text()

    assert "Local MVP ready" in report
    assert "Deployment gate" in report
    assert "Current limitations" in report
