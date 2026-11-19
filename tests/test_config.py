"""Tests for the configuration constants."""

from src import config


def test_metric_order_matches_model_metrics():
    for metrics in config.MODEL_METRICS.values():
        assert set(metrics) == set(config.METRIC_ORDER)


def test_metric_values_are_valid_percentages():
    for metrics in config.MODEL_METRICS.values():
        for value in metrics.values():
            assert 0.0 <= value <= 100.0


def test_history_columns_are_unique():
    assert len(config.HISTORY_COLUMNS) == len(set(config.HISTORY_COLUMNS))
