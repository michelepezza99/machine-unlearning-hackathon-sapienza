"""Smoke test delle principali trasformazioni di unlearning."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch

from machine_unlearning.data import load_challenge_data
from machine_unlearning.model import (
    build_model,
    build_model_from_artifact,
    load_model_artifact,
    model_state_to_cpu,
)
from machine_unlearning.training import compute_positive_class_weights
from machine_unlearning.unlearning import (
    apply_selective_fisher_dampening,
    build_fisher_mask,
    compute_diagonal_fisher,
    precompute_teacher_logits,
    repair_with_distillation_fixed,
)


def test_fisher_mask_dampening_and_repair(synthetic_data_dir: Path) -> None:
    device = torch.device("cpu")
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=11)
    artifact = load_model_artifact(synthetic_data_dir / "model_artifact")
    original_model = build_model_from_artifact(artifact, device=device)
    original_state = model_state_to_cpu(original_model)

    def model_builder(
        state_dict: Mapping[str, torch.Tensor] | None,
    ) -> torch.nn.Module:
        return build_model(
            artifact["architecture"], state_dict=state_dict, device=device
        )

    retain_fisher, _ = compute_diagonal_fisher(
        original_model,
        data.x_retain_train,
        data.y_retain_train,
        device=device,
        sample_size=4,
        batch_size=2,
        seed=11,
    )
    forget_fisher, _ = compute_diagonal_fisher(
        original_model,
        data.x_forget,
        data.y_forget,
        device=device,
        sample_size=4,
        batch_size=2,
        seed=12,
    )
    masks, _, metadata = build_fisher_mask(
        original_model,
        retain_fisher,
        forget_fisher,
        top_fraction=0.1,
        forget_absolute_quantile=0.25,
    )
    assert metadata["selected"] > 0

    dampened_model, dampening = apply_selective_fisher_dampening(
        model_builder,
        original_state,
        retain_fisher,
        forget_fisher,
        masks,
        device=device,
        minimum_factor=0.8,
        strength=1.0,
        ratio_power=1.0,
    )
    assert dampening["modified_total"] == metadata["selected"]
    protected_state = model_state_to_cpu(dampened_model)
    teacher_logits = precompute_teacher_logits(
        original_model,
        data.x_retain_train,
        device=device,
        batch_size=8,
    )
    weights = compute_positive_class_weights(data.y_retain_train, device=device)
    repaired_model, history = repair_with_distillation_fixed(
        dampened_model,
        retain_features=data.x_retain_train,
        retain_targets=data.y_retain_train,
        retain_teacher_logits=teacher_logits,
        original_state=original_state,
        masks=masks,
        device=device,
        seed=11,
        epochs=1,
        learning_rate=1e-4,
        weight_decay=1e-3,
        batch_size=8,
        positive_class_weights=weights,
        supervised_weight=1.0,
        distillation_weight=0.5,
        parameter_regularization_weight=1e-4,
        selected_parameter_weight=1.0,
        gradient_clip=1.0,
        freeze_selected=True,
    )
    repaired_state = model_state_to_cpu(repaired_model)
    for name, mask in masks.items():
        if bool(mask.any()):
            assert torch.equal(repaired_state[name][mask], protected_state[name][mask])
    assert len(history) == 1
