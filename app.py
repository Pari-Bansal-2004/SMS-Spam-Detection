"""Streamlit entry point for the SMS Spam Detection application."""

from __future__ import annotations

import streamlit as st

from src import config
from src.models import ModelArtifacts, ModelLoadError, load_artifacts
from src.styles import inject_css
from src.views import about, dashboard, detection, history_view

PAGES = {
    "Dashboard": dashboard.render,
    "Spam Detection": detection.render,
    "Prediction History": history_view.render,
    "About Project": about.render,
}


@st.cache_resource(show_spinner=False)
def get_artifacts() -> ModelArtifacts:
    """Load and cache the fitted models for the lifetime of the session."""
    return load_artifacts()


def main() -> None:
    st.set_page_config(
        page_title=config.APP_TITLE,
        page_icon=config.PAGE_ICON,
        layout="wide",
    )
    inject_css()

    try:
        artifacts = get_artifacts()
    except ModelLoadError as exc:
        st.error("Unable to load the saved models. Please check the 'models' folder.")
        st.exception(exc)
        st.stop()

    st.sidebar.title("📩 SMS Detector")
    selection = st.sidebar.radio("Navigation", list(PAGES))
    st.sidebar.markdown("---")
    st.sidebar.info(
        "MCA Minor Project\n\nMachine Learning Based SMS Classification"
    )

    PAGES[selection](artifacts)


if __name__ == "__main__":
    main()
