"""Prediction history page: view, download and clear past predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config
from ..history import clear_history, load_history
from ..models import ModelArtifacts


def render(artifacts: ModelArtifacts | None = None) -> None:
    """Render the prediction history page."""
    st.title("📋 Prediction History")

    rows = load_history()
    if not rows:
        st.info(
            "No prediction history available. Please test some messages first."
        )
        return

    frame = pd.DataFrame(rows, columns=config.HISTORY_COLUMNS)
    frame.insert(0, "No.", range(1, len(frame) + 1))

    st.metric("Total Messages Checked", len(frame))
    st.markdown("---")

    st.dataframe(frame, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download History (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )
    with col2:
        if st.button("Clear History", width="stretch"):
            clear_history()
            st.success("Prediction history cleared successfully!")
            st.rerun()
