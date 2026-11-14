from pathlib import Path


def test_repository_foundation_files_exist() -> None:
    root = Path(__file__).parents[1]
    assert (root / "README.md").is_file()
    assert (root / ".gitignore").is_file()
    assert (root / "pyproject.toml").is_file()
    assert (root / ".env.example").is_file()
