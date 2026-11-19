"""Spam detection page: classify a single message or a batch of messages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config
from ..history import append_history
from ..models import ModelArtifacts
from ..predictor import Prediction, classify, classify_many


def _render_prediction(title: str, prediction: Prediction) -> None:
    st.metric(title, prediction.label.upper())
    st.progress(
        prediction.confidence,
        text=f"Confidence: {prediction.confidence * 100:.1f}%",
    )


def _render_single(artifacts: ModelArtifacts) -> None:
    st.markdown("#### Enter your message")
    message = st.text_area(
        "Message",
        height=150,
        placeholder="Type your SMS message here...",
        label_visibility="collapsed",
    )

    if not st.button("Predict Message", type="primary", width="stretch"):
        return

    if not message.strip():
        st.warning("Please enter a message first.")
        return

    result = classify(message, artifacts)

    if result.verdict.label == config.SPAM:
        st.error("🚨 SPAM MESSAGE DETECTED")
    else:
        st.success("✅ HAM MESSAGE DETECTED")

    col1, col2 = st.columns(2)
    with col1:
        _render_prediction("Naive Bayes Prediction", result.naive_bayes)
    with col2:
        _render_prediction("SVM Prediction", result.svm)

    if not result.models_agree:
        st.warning("The models disagree on this message. Please review it manually.")

    append_history(message, result.naive_bayes.label, result.svm.label)
    st.caption("Prediction saved to history.")


def _parse_input_messages(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_uploaded_csv(file) -> tuple[list[str] | None, str]:
    try:
        frame = pd.read_csv(file)
    except Exception:
        return None, "Could not parse the uploaded file as CSV."

    if frame.empty:
        return None, "The uploaded CSV is empty."

    candidates = [
        col
        for col in frame.columns
        if any(k in col.lower() for k in ("message", "text", "sms", "content"))
    ]
    column = candidates[0] if candidates else frame.columns[0]
    messages = frame[column].dropna().astype(str).tolist()
    if not messages:
        return None, f"No messages found in column '{column}'."
    return messages, column


def _results_dataframe(messages: list[str], results) -> pd.DataFrame:
    rows = []
    for msg, result in zip(messages, results):
        rows.append(
            {
                "Message": msg,
                "Naive Bayes": result.naive_bayes.label.upper(),
                "SVM": result.svm.label.upper(),
                "Verdict": result.verdict.label.upper(),
                "Agreement": "Yes" if result.models_agree else "No",
            }
        )
    return pd.DataFrame(rows)


def _render_bulk(artifacts: ModelArtifacts) -> None:
    mode = st.radio(
        "Input method",
        options=["Paste messages", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
    )

    messages: list[str] | None = None

    if mode == "Paste messages":
        st.caption("Paste one SMS message per line.")
        text = st.text_area(
            "Messages",
            height=220,
            placeholder="Hey, are we meeting tomorrow?\nCongratulations! You won a prize. Call now.",
            label_visibility="collapsed",
        )
        if text.strip():
            messages = _parse_input_messages(text)
    else:
        uploaded = st.file_uploader(
            "Upload a CSV file",
            type=["csv"],
            help="The first text-like column (e.g. 'message', 'text') will be used.",
        )
        if uploaded is not None:
            messages, column = _parse_uploaded_csv(uploaded)
            if messages is None:
                st.error(column)
                return
            st.caption(f"Using column: **{column}** ({len(messages)} messages)")

    if messages is None or not messages:
        return

    st.info(f"{len(messages)} message(s) ready to classify.")

    if not st.button("Run Bulk Prediction", type="primary", width="stretch"):
        return

    with st.spinner("Classifying messages..."):
        results = classify_many(messages, artifacts)

    if not results:
        st.warning("No valid messages to classify.")
        return

    frame = _results_dataframe(messages, results)
    spam_count = int((frame["Verdict"] == "SPAM").sum())
    ham_count = len(frame) - spam_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Messages", len(frame))
    m2.metric("Spam Detected", spam_count, delta=f"{spam_count / len(frame) * 100:.1f}%")
    m3.metric("Ham Messages", ham_count, delta=f"{ham_count / len(frame) * 100:.1f}%")

    st.markdown("#### Results")
    st.dataframe(frame, width="stretch", hide_index=True)

    csv_data = frame.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Results (CSV)",
        data=csv_data,
        file_name="bulk_prediction_results.csv",
        mime="text/csv",
    )

    if st.button("Save all to prediction history"):
        for msg, result in zip(messages, results):
            append_history(msg, result.naive_bayes.label, result.svm.label)
        st.success(f"Saved {len(results)} predictions to history.")


def render(artifacts: ModelArtifacts | None = None) -> None:
    """Render the spam detection page."""
    st.title("🔍 SMS Spam Detection")
    st.write("Classify a single message or run predictions on a batch.")

    tab_single, tab_bulk = st.tabs(["Single Message", "Bulk Prediction"])
    with tab_single:
        _render_single(artifacts)
    with tab_bulk:
        _render_bulk(artifacts)
