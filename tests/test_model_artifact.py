"""Test di ricostruzione stretta dell'artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from machine_unlearning.metrics import predict_logits
from machine_unlearning.model import load_model_artifact, validate_artifact_payload


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
