"""Central configuration: file paths and project-wide constants."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
HISTORY_FILE = BASE_DIR / "prediction_history.csv"

TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
NB_MODEL_PATH = MODEL_DIR / "naive_bayes_model.pkl"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.pkl"

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
HAM = "ham"
SPAM = "spam"

# Model whose prediction is shown as the overall verdict.
PRIMARY_MODEL = "SVM"

HISTORY_COLUMNS = ["Message", "Naive Bayes", "SVM"]

METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Offline evaluation results on the held-out test split.
# Single source of truth for every metric rendered in the UI.
MODEL_METRICS: dict[str, dict[str, float]] = {
    "Naive Bayes": {
        "Accuracy": 95.26,
        "Precision": 100.00,
        "Recall": 62.60,
        "F1-Score": 77.00,
    },
    "Support Vector Machine (SVM)": {
        "Accuracy": 98.16,
        "Precision": 96.67,
        "Recall": 88.55,
        "F1-Score": 92.43,
    },
}

DATASET_RECORDS = 5169

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_TITLE = "SMS Spam Detection"
PAGE_ICON = "📩"
APP_SUBTITLE = "Machine Learning Based Message Classification System"
