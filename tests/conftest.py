"""Shared test setup."""
import pytest


@pytest.fixture(autouse=True)
def isolated_phrases_file(tmp_path, monkeypatch):
    """Point the default abbreviations file at an empty temp location, so a developer's own saved
    abbreviations (data/phrases.json) can never change what the tests see."""
    monkeypatch.setattr("app.cli.PHRASES_PATH", tmp_path / "default-phrases.json")
