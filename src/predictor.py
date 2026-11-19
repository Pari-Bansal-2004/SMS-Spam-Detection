"""Text classification using the fitted Naive Bayes and SVM models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .models import ModelArtifacts


@dataclass(frozen=True)
class Prediction:
    """A single model's label and confidence score (in ``[0, 1]``)."""

    label: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Combined predictions of both models for one message."""

    naive_bayes: Prediction
    svm: Prediction

    @property
    def verdict(self) -> Prediction:
        """The prediction of the primary model (SVM)."""
        return self.svm

    @property
    def models_agree(self) -> bool:
        """Whether both models returned the same label."""
        return self.naive_bayes.label == self.svm.label


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _predict_naive_bayes(model: Any, features: Any) -> Prediction:
    label = str(model.predict(features)[0])
    confidence = float(max(model.predict_proba(features)[0]))
    return Prediction(label=label, confidence=confidence)


def _predict_svm(model: Any, features: Any) -> Prediction:
    label = str(model.predict(features)[0])
    # LinearSVC exposes a signed distance to the hyperplane; map its magnitude
    # to a bounded, human-friendly confidence score.
    margin = abs(float(model.decision_function(features)[0]))
    return Prediction(label=label, confidence=_sigmoid(margin))


def classify(message: str, artifacts: ModelArtifacts) -> PredictionResult:
    """Classify a single SMS message with both models.

    Parameters
    ----------
    message:
        The raw SMS text to classify.
    artifacts:
        The loaded vectorizer and classifiers.

    Returns
    -------
    PredictionResult
        The Naive Bayes and SVM predictions for the message.

    Raises
    ------
    ValueError
        If ``message`` is empty or contains only whitespace.
    """
    if not message or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    features = artifacts.vectorizer.transform([message])
    return PredictionResult(
        naive_bayes=_predict_naive_bayes(artifacts.naive_bayes, features),
        svm=_predict_svm(artifacts.svm, features),
    )


def classify_many(messages: list[str], artifacts: ModelArtifacts) -> list[PredictionResult]:
    """Classify many messages in a single pass for efficiency.

    The TF-IDF vectorizer and both classifiers are applied once to the full
    batch instead of per-message, which is significantly faster for bulk input.
    Empty or whitespace-only messages are skipped.

    Parameters
    ----------
    messages:
        The raw SMS texts to classify.
    artifacts:
        The loaded vectorizer and classifiers.

    Returns
    -------
    list[PredictionResult]
        One prediction per non-empty message, in input order.
    """
    valid = [msg for msg in messages if msg and msg.strip()]
    if not valid:
        return []

    features = artifacts.vectorizer.transform(valid)

    nb_labels = artifacts.naive_bayes.predict(features)
    nb_confidences = artifacts.naive_bayes.predict_proba(features).max(axis=1)

    svm_labels = artifacts.svm.predict(features)
    svm_margins = artifacts.svm.decision_function(features)

    return [
        PredictionResult(
            naive_bayes=Prediction(
                label=str(nb_labels[i]),
                confidence=float(nb_confidences[i]),
            ),
            svm=Prediction(
                label=str(svm_labels[i]),
                confidence=_sigmoid(abs(float(svm_margins[i]))),
            ),
        )
        for i in range(len(valid))
    ]
