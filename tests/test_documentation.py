from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_product_documentation_exists() -> None:
    assert (ROOT / "docs" / "architecture.md").is_file()
    assert (ROOT / "docs" / "api.md").is_file()
    assert "Quick start" in (ROOT / "README.md").read_text(encoding="utf-8")
