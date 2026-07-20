"""Test manualmente verificabili delle metriche autorevoli."""

from __future__ import annotations

import math

import numpy as np
import pytest

from machine_unlearning.metrics import (
    binary_cross_entropy_from_logits,
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
