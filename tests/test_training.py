"""Test mirati per batching deterministico e checkpoint addestrati."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

import machine_unlearning.training as training_module
from machine_unlearning.model import DynamicMLP, model_state_to_cpu
from machine_unlearning.training import (
    MAX_RANDOM_SEED,
    make_data_loader,
    seed_everything,
    train_fixed_epochs,
    train_with_early_stopping,
    validate_seed,
)


def _training_arrays(sample_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    features = np.arange(sample_count * 2, dtype=np.float32).reshape(sample_count, 2)
    targets = np.column_stack(
        [
            np.arange(sample_count) % 2,
            (np.arange(sample_count) + 1) % 2,
        ]
    ).astype(np.float32)
    return features, targets


def test_seed_range_accepts_both_boundaries() -> None:
    assert validate_seed(0) == 0
    assert validate_seed(MAX_RANDOM_SEED) == MAX_RANDOM_SEED


@pytest.mark.parametrize(
    ("invalid_seed", "exception_type"),
    [
        (-1, ValueError),
        (MAX_RANDOM_SEED + 1, ValueError),
        (True, TypeError),
    ],
)
def test_seed_everything_rejects_unsupported_values(
    invalid_seed: int, exception_type: type[Exception]
) -> None:
    with pytest.raises(exception_type, match="seed"):
        seed_everything(invalid_seed)


@pytest.mark.parametrize(
    ("sample_count", "requested_batch_size", "expected_batch_sizes"),
    [
        (2, 1, [2]),
        (3, 2, [3]),
        (7, 3, [3, 4]),
        (8, 4, [4, 4]),
        (5, 20, [5]),
    ],
)
def test_batchnorm_loader_never_emits_singleton(
    sample_count: int,
    requested_batch_size: int,
    expected_batch_sizes: list[int],
) -> None:
    features, targets = _training_arrays(sample_count)
    loader = make_data_loader(
        features,
        targets,
        batch_size=requested_batch_size,
        shuffle=False,
        seed=7,
        device=torch.device("cpu"),
        batchnorm_training=True,
    )

    observed_batch_sizes = [len(feature_batch) for feature_batch, _ in loader]
    assert observed_batch_sizes == expected_batch_sizes
    assert sum(observed_batch_sizes) == sample_count
    assert min(observed_batch_sizes) >= 2


@pytest.mark.parametrize("batch_size", [0, -3])
def test_loader_rejects_nonpositive_batch_size(batch_size: int) -> None:
    features, targets = _training_arrays()
    with pytest.raises(ValueError, match="batch_size deve essere positivo"):
        make_data_loader(
            features,
            targets,
            batch_size=batch_size,
            shuffle=False,
            seed=7,
            device=torch.device("cpu"),
        )


def test_loader_shuffle_is_deterministic_for_fixed_seed() -> None:
    features, targets = _training_arrays(11)

    def shuffled_rows() -> torch.Tensor:
        loader = make_data_loader(
            features,
            targets,
            batch_size=3,
            shuffle=True,
            seed=23,
            device=torch.device("cpu"),
            batchnorm_training=True,
        )
        return torch.cat([feature_batch for feature_batch, _ in loader])

    assert torch.equal(shuffled_rows(), shuffled_rows())


def test_fixed_training_rejects_zero_epochs() -> None:
    features, targets = _training_arrays()
    model = DynamicMLP(input_dim=2, hidden_layers=[4], num_outputs=2)
    with pytest.raises(ValueError, match="almeno 1"):
        train_fixed_epochs(
            model,
            features,
            targets,
            device=torch.device("cpu"),
            seed=7,
            epochs=0,
            batch_size=4,
            learning_rate=0.01,
            weight_decay=0.0,
            optimizer_name="adam",
            positive_class_weights=torch.ones(2),
        )


def test_early_stopping_selects_first_trained_epoch_when_initial_is_better(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, targets = _training_arrays()
    model = DynamicMLP(input_dim=2, hidden_layers=[4], num_outputs=2)
    initial_state = model_state_to_cpu(model)
    precisions: Iterator[float] = iter([0.9, 0.2, 0.1])

    def fake_evaluate_model(*args: object, **kwargs: object) -> dict[str, float]:
        return {
            "precision_at_10": next(precisions),
            "bce_from_logits": 1.0,
        }

    monkeypatch.setattr(training_module, "evaluate_model", fake_evaluate_model)
    result = train_with_early_stopping(
        model,
        features,
        targets,
        features,
        targets,
        device=torch.device("cpu"),
        seed=7,
        max_epochs=2,
        patience=2,
        batch_size=4,
        evaluation_batch_size=4,
        learning_rate=0.01,
        weight_decay=0.0,
        optimizer_name="adam",
        positive_class_weights=torch.ones(2),
    )

    assert result.best_epoch == 1
    assert result.best_precision_at_10 == pytest.approx(0.2)
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in result.best_state_dict.values()
        if tensor.is_floating_point()
    )
    assert any(
        not torch.equal(result.best_state_dict[name], initial_state[name])
        for name in initial_state
        if initial_state[name].is_floating_point()
    )


@pytest.mark.parametrize(
    ("max_epochs", "patience", "message"),
    [(0, 1, "max_epochs"), (1, 0, "patience")],
)
def test_early_stopping_rejects_invalid_settings(
    max_epochs: int, patience: int, message: str
) -> None:
    features, targets = _training_arrays()
    model = DynamicMLP(input_dim=2, hidden_layers=[4], num_outputs=2)
    with pytest.raises(ValueError, match=message):
        train_with_early_stopping(
            model,
            features,
            targets,
            features,
            targets,
            device=torch.device("cpu"),
            seed=7,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=4,
            evaluation_batch_size=4,
            learning_rate=0.01,
            weight_decay=0.0,
            optimizer_name="adam",
            positive_class_weights=torch.ones(2),
        )
