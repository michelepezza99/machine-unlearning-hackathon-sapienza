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
    recalibrate_batchnorm,
    repair_with_distillation_fixed,
    selective_gradient_ascent,
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
    assert all(bool(torch.isfinite(values).all()) for values in retain_fisher.values())
    assert all(bool(torch.isfinite(values).all()) for values in forget_fisher.values())
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
    changed_selected = 0
    for name, mask in masks.items():
        assert torch.equal(protected_state[name][~mask], original_state[name][~mask])
        if bool(mask.any()):
            changed_selected += int(
                torch.count_nonzero(
                    protected_state[name][mask] - original_state[name][mask]
                ).item()
            )
    assert changed_selected > 0
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
    assert repaired_model.training is False
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in repaired_state.values()
        if tensor.is_floating_point()
    )


def test_gradient_ascent_updates_only_selected_elements(
    synthetic_data_dir: Path,
) -> None:
    device = torch.device("cpu")
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=11)
    artifact = load_model_artifact(synthetic_data_dir / "model_artifact")
    model = build_model_from_artifact(artifact, device=device)
    before = model_state_to_cpu(model)
    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool)
        for name, parameter in model.named_parameters()
    }
    selected_name = next(
        name for name, parameter in model.named_parameters() if parameter.ndim >= 2
    )
    masks[selected_name].view(-1)[:4] = True
    teacher_logits = precompute_teacher_logits(
        model,
        data.x_retain_train,
        device=device,
        batch_size=8,
    )

    updated_model, history = selective_gradient_ascent(
        model,
        forget_features=data.x_forget,
        forget_targets=data.y_forget,
        retain_features=data.x_retain_train,
        retain_teacher_logits=teacher_logits,
        masks=masks,
        device=device,
        seed=11,
        learning_rate=1e-3,
        steps=2,
        batch_size=4,
        retain_distillation_weight=0.5,
        gradient_clip=10.0,
    )
    after = model_state_to_cpu(updated_model)
    assert len(history) == 2
    assert updated_model.training is False
    for name, mask in masks.items():
        assert torch.equal(after[name][~mask], before[name][~mask])
    assert not torch.equal(
        after[selected_name][masks[selected_name]],
        before[selected_name][masks[selected_name]],
    )


def test_batchnorm_recalibration_is_deterministic_and_finite(
    synthetic_data_dir: Path,
) -> None:
    device = torch.device("cpu")
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=11)
    artifact = load_model_artifact(synthetic_data_dir / "model_artifact")
    first = build_model_from_artifact(artifact, device=device)
    second = build_model_from_artifact(artifact, device=device)

    recalibrate_batchnorm(
        first,
        data.x_retain_train,
        device=device,
        batch_size=7,
    )
    recalibrate_batchnorm(
        second,
        data.x_retain_train,
        device=device,
        batch_size=7,
    )

    first_state = model_state_to_cpu(first)
    second_state = model_state_to_cpu(second)
    assert first.training is False
    assert second.training is False
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)
    assert all(
        bool(torch.isfinite(tensor).all())
        for tensor in first_state.values()
        if tensor.is_floating_point()
    )
    batch_counters = [
        tensor
        for name, tensor in first_state.items()
        if name.endswith("num_batches_tracked")
    ]
    assert batch_counters and all(int(counter.item()) > 0 for counter in batch_counters)
