"""Test di ricostruzione stretta dell'artifact."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from machine_unlearning.metrics import predict_logits
from machine_unlearning.model import (
    REQUIRED_ARTIFACT_KEYS,
    load_model_artifact,
    validate_artifact_payload,
)


def test_artifact_loads_strictly_on_cpu(synthetic_data_dir: Path) -> None:
    payload = load_model_artifact(synthetic_data_dir / "model_artifact")
    model = validate_artifact_payload(payload)
    assert all(tensor.device.type == "cpu" for tensor in payload["state_dict"].values())
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in payload["state_dict"].values()
        if tensor.is_floating_point()
    )
    logits = predict_logits(
        model,
        np.zeros((2, 4), dtype=np.float32),
        device=torch.device("cpu"),
        batch_size=2,
    )
    assert logits.shape == (2, 12)


@pytest.mark.parametrize("missing_key", sorted(REQUIRED_ARTIFACT_KEYS))
def test_artifact_rejects_each_missing_required_key(
    synthetic_data_dir: Path,
    missing_key: str,
) -> None:
    payload = load_model_artifact(synthetic_data_dir / "model_artifact")
    payload.pop(missing_key)
    with pytest.raises(KeyError, match="Artifact incompleto"):
        validate_artifact_payload(payload)


def test_artifact_rejects_non_finite_parameters(synthetic_data_dir: Path) -> None:
    payload = load_model_artifact(synthetic_data_dir / "model_artifact")
    corrupted = deepcopy(payload)
    floating_name = next(
        name
        for name, tensor in corrupted["state_dict"].items()
        if tensor.is_floating_point()
    )
    corrupted["state_dict"][floating_name].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="non finiti"):
        validate_artifact_payload(corrupted)


def test_artifact_uses_strict_state_reconstruction(synthetic_data_dir: Path) -> None:
    payload = load_model_artifact(synthetic_data_dir / "model_artifact")
    missing_parameter = next(iter(payload["state_dict"]))
    payload["state_dict"].pop(missing_parameter)
    with pytest.raises(RuntimeError, match="Missing key"):
        validate_artifact_payload(payload)
