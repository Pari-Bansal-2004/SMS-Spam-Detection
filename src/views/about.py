"""About page: project description and technologies."""

from __future__ import annotations

import streamlit as st

from .. import config
from ..models import ModelArtifacts


def render(artifacts: ModelArtifacts | None = None) -> None:
    """Render the about page."""
    st.title("ℹ️ About Project")
    st.subheader("SMS Spam Detection System")

    st.write(
        "The SMS Spam Detection System is a machine learning application "
        "designed to identify unwanted and fraudulent SMS messages."
    )

    st.markdown(
        """
        **Technologies Used:**
        - Python
        - Machine Learning
        - Scikit-learn
        - TF-IDF Vectorization
        - Streamlit
        - Pandas

        **Algorithms:**
        - Multinomial Naive Bayes
        - Support Vector Machine (SVM)

        **Dataset:** SMS Spam Collection Dataset

        **Primary Model:** Support Vector Machine (SVM)
        """
    )

    st.markdown("---")
    st.caption("MCA Minor Project | SMS Spam Detection System")
