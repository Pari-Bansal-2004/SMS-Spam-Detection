"""Loading of the persisted TF-IDF vectorizer and classifier artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib

from . import config


@dataclass(frozen=True)
class ModelArtifacts:
    """Bundle of the fitted vectorizer and both trained classifiers."""

    vectorizer: Any
    naive_bayes: Any
    svm: Any


class ModelLoadError(RuntimeError):
    """Raised when the persisted model files cannot be loaded."""


def load_artifacts() -> ModelArtifacts:
    """Load the fitted artifacts from disk.

    Parameters
    ----------
    None

    Returns
    -------
    ModelArtifacts
        The loaded vectorizer and classifiers.

    Raises
    ------
    ModelLoadError
        If any expected model file is missing or cannot be deserialized.
    """
    try:
        return ModelArtifacts(
            vectorizer=joblib.load(config.TFIDF_PATH),
            naive_bayes=joblib.load(config.NB_MODEL_PATH),
            svm=joblib.load(config.SVM_MODEL_PATH),
        )
    except Exception as exc:  # noqa: BLE001 - normalize any load failure
        raise ModelLoadError(str(exc)) from exc
