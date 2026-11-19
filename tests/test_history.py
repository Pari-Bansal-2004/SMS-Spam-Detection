"""Tests for prediction-history persistence."""

import pytest

from src import config, history


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Redirect history writes to a temporary file for every test."""
    target = tmp_path / "history.csv"
    monkeypatch.setattr(config, "HISTORY_FILE", target)
    return target


def test_load_missing_history_returns_empty_list():
    assert history.load_history() == []


def test_append_then_load_roundtrip():
    history.append_history("hello there", "ham", "ham")
    rows = history.load_history()

    assert len(rows) == 1
    assert rows[0]["Message"] == "hello there"
    assert rows[0]["Naive Bayes"] == "HAM"
    assert rows[0]["SVM"] == "HAM"


def test_labels_are_uppercased_and_multiple_rows_kept():
    history.append_history("first", "ham", "spam")
    history.append_history("second", "spam", "spam")
    rows = history.load_history()

    assert [row["Message"] for row in rows] == ["first", "second"]
    assert rows[0]["SVM"] == "SPAM"
    assert len(rows) == 2


def test_clear_history_leaves_only_header():
    history.append_history("something", "spam", "spam")
    history.clear_history()

    assert history.load_history() == []
    assert config.HISTORY_FILE.exists()
    assert config.HISTORY_FILE.read_text(encoding="utf-8").strip() == (
        ",".join(config.HISTORY_COLUMNS)
    )
