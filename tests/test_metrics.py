"""Test manualmente verificabili delle metriche autorevoli."""

from __future__ import annotations

import math

import numpy as np
import pytest

from machine_unlearning.metrics import (
    binary_cross_entropy_from_logits,
    local_multi_objective_score,
    precision_at_k,
)


def test_precision_at_10_manual_example() -> None:
    logits = np.vstack([np.arange(12), np.arange(12)[::-1]]).astype(np.float32)
    targets = np.zeros((2, 12), dtype=np.float32)
    targets[0, [0, 10, 11]] = 1.0
    targets[1, [0, 1, 11]] = 1.0
    assert precision_at_k(logits, targets, k=10) == pytest.approx(0.2)


def test_bce_for_zero_logits_is_log_two() -> None:
    logits = np.zeros((2, 3), dtype=np.float32)
    targets = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    assert math.isclose(
        binary_cross_entropy_from_logits(logits, targets),
        math.log(2.0),
        rel_tol=1e-6,
    )


@pytest.mark.parametrize(
    ("logit_shape", "target_shape"),
    [((2, 3), (2, 2)), ((2, 3), (6,)), ((0, 3), (0, 3))],
)
def test_metrics_reject_invalid_shapes(
    logit_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
) -> None:
    logits = np.zeros(logit_shape, dtype=np.float32)
    targets = np.zeros(target_shape, dtype=np.float32)
    with pytest.raises(ValueError):
        precision_at_k(logits, targets, k=1)
    with pytest.raises(ValueError):
        binary_cross_entropy_from_logits(logits, targets)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_metrics_reject_non_finite_values(invalid_value: float) -> None:
    logits = np.zeros((2, 3), dtype=np.float32)
    targets = np.zeros((2, 3), dtype=np.float32)
    logits[0, 0] = invalid_value
    with pytest.raises(ValueError, match="non finiti"):
        precision_at_k(logits, targets, k=1)
    with pytest.raises(ValueError, match="non finiti"):
        binary_cross_entropy_from_logits(logits, targets)


def test_local_search_score_is_deterministic_and_finite() -> None:
    first = local_multi_objective_score(
        precision_at_10_value=0.4,
        local_privacy_proxy=0.6,
        execution_time_seconds=2.0,
        retraining_time_seconds=5.0,
    )
    second = local_multi_objective_score(
        precision_at_10_value=0.4,
        local_privacy_proxy=0.6,
        execution_time_seconds=2.0,
        retraining_time_seconds=5.0,
    )
    assert first == second
    assert all(math.isfinite(value) for value in first)


def test_local_search_score_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError, match="non finiti"):
        local_multi_objective_score(
            precision_at_10_value=0.4,
            local_privacy_proxy=float("nan"),
            execution_time_seconds=2.0,
            retraining_time_seconds=5.0,
        )
