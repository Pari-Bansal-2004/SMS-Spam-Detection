"""End-to-end smoke tests that execute the Streamlit app headlessly."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = Path(__file__).resolve().parent.parent / "app.py"
PAGES = ["Dashboard", "Spam Detection", "Prediction History", "About Project"]


@pytest.fixture(scope="module")
def isolated_history(tmp_path_factory):
    """Point the app at a throwaway history file for the whole module."""
    target = tmp_path_factory.mktemp("data") / "history.csv"
    original = config.HISTORY_FILE
    config.HISTORY_FILE = target
    yield target
    config.HISTORY_FILE = original


@pytest.fixture(scope="module")
def app(isolated_history):
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception


def test_all_navigation_pages_render(app):
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page} page raised an exception"


def test_spam_message_is_classified(app):
    app.sidebar.radio[0].set_value("Spam Detection").run()
    app.text_area[0].set_value(
        "Congratulations! You have won a free prize. Call now to claim your reward!"
    ).run()
    app.button[0].click().run()

    assert not app.exception
    assert any("SPAM MESSAGE DETECTED" in block.value for block in app.error)
