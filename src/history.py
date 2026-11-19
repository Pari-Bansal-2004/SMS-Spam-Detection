"""Persistence of prediction history to a CSV file."""

from __future__ import annotations

import csv

from . import config


def load_history() -> list[dict[str, str]]:
    """Return all saved predictions as a list of row dictionaries."""
    file = config.HISTORY_FILE
    if not file.exists() or file.stat().st_size == 0:
        return []

    with file.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_history(message: str, naive_bayes_label: str, svm_label: str) -> None:
    """Append a single prediction to the history file, writing a header first."""
    file = config.HISTORY_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file.exists() or file.stat().st_size == 0

    with file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(config.HISTORY_COLUMNS)
        writer.writerow([message, naive_bayes_label.upper(), svm_label.upper()])


def clear_history() -> None:
    """Reset the history file so that it contains only the header row."""
    file = config.HISTORY_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(config.HISTORY_COLUMNS)
