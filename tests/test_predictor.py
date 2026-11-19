"""Tests for the message classifier."""

import pytest

from src import config
from src.models import load_artifacts
from src.predictor import classify, classify_many

HAM_MESSAGE = "Hey, are we still meeting for lunch tomorrow?"
SPAM_MESSAGE = (
    "Congratulations! You have won a free prize. "
    "Call now to claim your reward!"
)


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts()


def test_returns_a_label_from_each_model(artifacts):
    result = classify(HAM_MESSAGE, artifacts)
    assert result.naive_bayes.label in {config.HAM, config.SPAM}
    assert result.svm.label in {config.HAM, config.SPAM}


def test_confidences_are_bounded(artifacts):
    result = classify(SPAM_MESSAGE, artifacts)
    assert 0.0 <= result.naive_bayes.confidence <= 1.0
    assert 0.0 <= result.svm.confidence <= 1.0


def test_obvious_spam_is_flagged(artifacts):
    result = classify(SPAM_MESSAGE, artifacts)
    assert result.svm.label == config.SPAM


def test_verdict_uses_primary_model(artifacts):
    result = classify(HAM_MESSAGE, artifacts)
    assert result.verdict is result.svm


def test_agree_flag_reflects_labels(artifacts):
    result = classify(SPAM_MESSAGE, artifacts)
    assert result.models_agree == (
        result.naive_bayes.label == result.svm.label
    )


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_blank_messages_are_rejected(artifacts, message):
    with pytest.raises(ValueError):
        classify(message, artifacts)


def test_classify_many_returns_one_result_per_message(artifacts):
    messages = [HAM_MESSAGE, SPAM_MESSAGE, "URGENT! Claim your free gift now."]
    results = classify_many(messages, artifacts)

    assert len(results) == len(messages)
    for result in results:
        assert result.naive_bayes.label in {config.HAM, config.SPAM}
        assert result.svm.label in {config.HAM, config.SPAM}


def test_classify_many_skips_blank_messages(artifacts):
    messages = [HAM_MESSAGE, "   ", "", SPAM_MESSAGE, "\t"]
    results = classify_many(messages, artifacts)
    assert len(results) == 2


def test_classify_many_empty_input_returns_empty(artifacts):
    assert classify_many([], artifacts) == []
    assert classify_many(["", "  "], artifacts) == []


def test_classify_many_matches_single_classification(artifacts):
    messages = [HAM_MESSAGE, SPAM_MESSAGE]
    batch = classify_many(messages, artifacts)

    for message, batch_result in zip(messages, batch):
        single = classify(message, artifacts)
        assert single.svm.label == batch_result.svm.label
        assert single.naive_bayes.label == batch_result.naive_bayes.label
