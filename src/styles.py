"""Global application styling."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
.main {
    background-color: #f5f7fb;
}

/* Clean, light metric cards with subtle shadow */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 18px 20px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
}

div[data-testid="stMetricLabel"] p {
    color: #64748b !important;
    font-size: 13px !important;
}

div[data-testid="stMetricValue"] {
    color: #0f172a !important;
}

div[data-testid="stMetricDelta"] {
    color: #10b981 !important;
}

.dashboard-title {
    font-size: 34px;
    font-weight: 800;
    color: #0f172a;
}

.dashboard-subtitle {
    color: #64748b;
    font-size: 16px;
    margin-bottom: 20px;
}

.chart-card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
    margin-bottom: 20px;
    color: #334155;
}

.chart-heading {
    font-size: 21px;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 20px;
}

.bar-label {
    font-size: 14px;
    font-weight: 600;
    color: #334155;
    margin-bottom: 6px;
}

.bar-track {
    width: 100%;
    height: 15px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 16px;
}

.bar-fill-nb {
    height: 100%;
    background: #3b82f6;
    border-radius: 10px;
}

.bar-fill-svm {
    height: 100%;
    background: #10b981;
    border-radius: 10px;
}

.legend {
    display: flex;
    gap: 22px;
    margin-bottom: 18px;
    color: #475569;
    font-size: 13px;
}

.legend span {
    display: flex;
    align-items: center;
    gap: 6px;
}

.blue-dot, .green-dot {
    width: 11px;
    height: 11px;
    border-radius: 3px;
    display: inline-block;
}

.blue-dot {
    background: #3b82f6;
}

.green-dot {
    background: #10b981;
}

.metric-value {
    font-size: 16px;
    font-weight: 700;
    color: #0f172a;
    float: right;
}

.performance-row {
    margin-bottom: 22px;
}

/* Clean dataframes */
div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Subtle info/success cards */
div[data-testid="stAlert"] {
    border-radius: 10px;
}
</style>
"""


def inject_css() -> None:
    """Inject the global stylesheet into the current page."""
    st.markdown(CSS, unsafe_allow_html=True)
