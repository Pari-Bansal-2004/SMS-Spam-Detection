"""Dashboard page: model metrics and project overview."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from .. import config
from ..models import ModelArtifacts

_NB = "Naive Bayes"
_SVM = "Support Vector Machine (SVM)"
_COLOR_NB = "#3b82f6"
_COLOR_SVM = "#10b981"


def _metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": metric,
                _NB: config.MODEL_METRICS[_NB][metric],
                _SVM: config.MODEL_METRICS[_SVM][metric],
            }
            for metric in config.METRIC_ORDER
        ]
    )


def _accuracy_bar() -> alt.Chart:
    data = pd.DataFrame(
        [
            {"Model": "Naive Bayes", "Accuracy": config.MODEL_METRICS[_NB]["Accuracy"]},
            {"Model": "SVM", "Accuracy": config.MODEL_METRICS[_SVM]["Accuracy"]},
        ]
    )
    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Model:N", axis=alt.Axis(labelAngle=0, title=None)),
            y=alt.Y("Accuracy:Q", scale=alt.Scale(domain=[0, 100]), title="Accuracy (%)"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=["Naive Bayes", "SVM"], range=[_COLOR_NB, _COLOR_SVM]),
                legend=None,
            ),
            tooltip=["Model:N", alt.Tooltip("Accuracy:Q", format=".2f")],
        )
        .properties(height=320)
    )
    text = bars.mark_text(align="center", dy=-10, fontWeight="bold").encode(
        text=alt.Text("Accuracy:Q", format=".2f")
    )
    return bars + text


def _grouped_metrics_bar() -> alt.Chart:
    frame = _metrics_frame().melt(
        id_vars="Metric", var_name="Model", value_name="Score"
    )
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N", axis=alt.Axis(labelAngle=0), title=None),
            xOffset="Model:N",
            y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]), title="Score (%)"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=[_NB, _SVM], range=[_COLOR_NB, _COLOR_SVM]),
                title=None,
            ),
            tooltip=[
                "Metric:N",
                "Model:N",
                alt.Tooltip("Score:Q", format=".2f"),
            ],
        )
        .properties(height=360)
    )


def _performance_line() -> alt.Chart:
    frame = _metrics_frame().melt(
        id_vars="Metric", var_name="Model", value_name="Score"
    )
    return (
        alt.Chart(frame)
        .mark_line(point=alt.OverlayMarkDef(size=80))
        .encode(
            x=alt.X(
                "Metric:N",
                sort=config.METRIC_ORDER,
                axis=alt.Axis(labelAngle=0, title=None),
            ),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="Score (%)",
            ),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(domain=[_NB, _SVM], range=[_COLOR_NB, _COLOR_SVM]),
                title=None,
            ),
            strokeWidth=alt.value(3),
            tooltip=[
                "Metric:N",
                "Model:N",
                alt.Tooltip("Score:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )


def render(artifacts: ModelArtifacts | None = None) -> None:
    """Render the dashboard page."""
    nb = config.MODEL_METRICS[_NB]
    svm = config.MODEL_METRICS[_SVM]

    st.markdown(
        '<div class="dashboard-title">SMS Spam Detection Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="dashboard-subtitle">{config.APP_SUBTITLE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Naive Bayes Accuracy", f"{nb['Accuracy']:.2f}%")
    c2.metric("SVM Accuracy", f"{svm['Accuracy']:.2f}%")
    c3.metric("Dataset Records", f"{config.DATASET_RECORDS:,}")
    c4.metric("Primary Model", config.PRIMARY_MODEL)

    st.markdown("---")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Model Accuracy")
        st.altair_chart(_accuracy_bar(), use_container_width=True)
    with right:
        st.markdown("#### Performance Comparison")
        st.altair_chart(_performance_line(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Detailed Metrics")
    st.altair_chart(_grouped_metrics_bar(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Metrics Summary")
    st.dataframe(
        _metrics_frame().set_index("Metric").style.format("{:.2f}%"),
        width="stretch",
    )

    st.markdown("---")
    st.markdown("### 🤖 Algorithms Used")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "**Multinomial Naive Bayes**\n\n"
            f"Accuracy: {nb['Accuracy']:.2f}%\n\n"
            "A probabilistic machine learning algorithm used for text "
            "classification."
        )
    with col2:
        st.success(
            "**Support Vector Machine (SVM)**\n\n"
            f"Accuracy: {svm['Accuracy']:.2f}%\n\n"
            "A supervised learning algorithm used to classify SMS messages."
        )

    st.markdown("---")
    st.markdown("### 📌 Project Overview")
    st.write(
        "This project uses machine learning to classify SMS messages into two "
        "categories: HAM (normal) and SPAM (unwanted). TF-IDF converts text "
        "messages into numerical features before classification. Multinomial "
        "Naive Bayes and Support Vector Machine algorithms are used for "
        "prediction."
    )
