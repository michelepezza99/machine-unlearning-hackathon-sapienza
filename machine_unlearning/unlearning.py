"""Fisher, dampening selettivo e repair per il machine unlearning."""

from __future__ import annotations

import gc
import time
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .metrics import (
    ReferencePrivacyProxy,
    evaluate_unlearning_candidate,
    predict_logits,
)
from .model import assert_finite_state, model_state_to_cpu
from .training import compute_positive_class_weights, make_data_loader, seed_everything


ModelBuilder = Callable[[Mapping[str, torch.Tensor] | None], torch.nn.Module]


@dataclass
class HybridUnlearningResult:
    """Risultato della riesecuzione fissa del metodo ibrido."""

    model: torch.nn.Module
    state_dict: dict[str, torch.Tensor]
    execution_time_seconds: float
    metadata: dict[str, Any]
    repair_history: pd.DataFrame
    gradient_ascent_history: pd.DataFrame


def release_memory() -> None:
    """Richiede il garbage collection e libera la cache CUDA disponibile.

    La funzione non pretende di eliminare riferimenti posseduti dal chiamante:
    questi vanno rimossi esplicitamente prima di invocarla.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_indices(sample_count: int, requested: int | None, seed: int) -> np.ndarray:
    if sample_count <= 0:
        raise ValueError("Non possiamo campionare da un dataset vuoto.")
    if requested is None or requested >= sample_count:
        return np.arange(sample_count, dtype=np.int64)
    if requested <= 0:
        raise ValueError("La dimensione del campione Fisher deve essere positiva.")
    generator = np.random.default_rng(seed)
    return np.sort(
        generator.choice(sample_count, size=requested, replace=False)
    ).astype(np.int64)


def _eligible_parameter_names(
    model: torch.nn.Module,
    *,
    include_bias: bool,
    include_batchnorm_affine: bool,
) -> set[str]:
    """Individua i soli parametri trainabili ammessi alla selezione Fisher."""
    batchnorm_parameters: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            if module.weight is not None:
                batchnorm_parameters.add(prefix + "weight")
            if module.bias is not None:
                batchnorm_parameters.add(prefix + "bias")

    eligible: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not include_bias and parameter.ndim < 2:
            continue
        if not include_batchnorm_affine and name in batchnorm_parameters:
            continue
        eligible.add(name)
    if not eligible:
        raise ValueError("Nessun parametro e' idoneo alla selezione Fisher.")
    return eligible


def compute_diagonal_fisher(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    sample_size: int | None,
    batch_size: int,
    seed: int,
    positive_class_weights: torch.Tensor | None = None,
    include_bias: bool = False,
    include_batchnorm_affine: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Calcola la media dei gradienti per-esempio al quadrato.

    Manteniamo il modello in `eval()` per non alterare le statistiche BatchNorm.
    I buffer non entrano nella Fisher: restituiamo esclusivamente i parametri
    idonei, normalizzati sul numero effettivo di esempi processati.
    """
    if len(features) != len(targets):
        raise ValueError("Feature e target Fisher non sono allineate.")
    indices = _sample_indices(len(features), sample_size, seed)
    sampled_features = features[indices]
    sampled_targets = targets[indices]

    fisher_model = deepcopy(model).to(device)
    fisher_model.eval()
    eligible = _eligible_parameter_names(
        fisher_model,
        include_bias=include_bias,
        include_batchnorm_affine=include_batchnorm_affine,
    )
    parameters = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in fisher_model.named_parameters()
    }
    buffers = {
        name: buffer.detach().clone() for name, buffer in fisher_model.named_buffers()
    }
    accumulator = {
        name: torch.zeros_like(parameters[name], dtype=torch.float64, device=device)
        for name in eligible
    }
    class_weights = (
        positive_class_weights.detach().to(device=device, dtype=torch.float32)
        if positive_class_weights is not None
        else None
    )

    def single_example_loss(
        functional_parameters: dict[str, torch.Tensor],
        functional_buffers: dict[str, torch.Tensor],
        example_features: torch.Tensor,
        example_targets: torch.Tensor,
    ) -> torch.Tensor:
        logits = functional_call(
            fisher_model,
            (functional_parameters, functional_buffers),
            (example_features.unsqueeze(0),),
        ).squeeze(0)
        return F.binary_cross_entropy_with_logits(
            logits,
            example_targets,
            pos_weight=class_weights,
            reduction="mean",
        )

    per_example_gradient = vmap(
        grad(single_example_loss),
        in_dims=(None, None, 0, 0),
        randomness="different",
    )
    loader = make_data_loader(
        sampled_features,
        sampled_targets,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    processed = 0
    start = time.perf_counter()
    for feature_batch, target_batch in loader:
        feature_batch = feature_batch.to(device, non_blocking=device.type == "cuda")
        target_batch = target_batch.to(device, non_blocking=device.type == "cuda")
        gradients = per_example_gradient(
            parameters, buffers, feature_batch, target_batch
        )
        for name in eligible:
            squared_sum = gradients[name].detach().to(torch.float64).square().sum(dim=0)
            if not bool(torch.isfinite(squared_sum).all()):
                raise FloatingPointError(f"Fisher non finita per {name}.")
            accumulator[name].add_(squared_sum)
        processed += len(feature_batch)
        del gradients
    elapsed = time.perf_counter() - start

    if processed != len(sampled_features):
        raise RuntimeError(
            f"Calcolo Fisher incompleto: {processed}/{len(sampled_features)} esempi."
        )
    fisher = {
        name: (values / processed).to(dtype=torch.float32, device="cpu")
        for name, values in accumulator.items()
    }
    for name, values in fisher.items():
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError(f"Fisher normalizzata non finita per {name}.")

    metadata = {
        "sample_size": int(processed),
        "batch_size": int(batch_size),
        "elapsed_seconds": float(elapsed),
        "eligible_parameter_names": sorted(eligible),
        "model_mode": "eval",
        "normalization": "mean_of_squared_per_example_gradients",
        "uses_positive_class_weights": class_weights is not None,
    }
    release_memory()
    return fisher, metadata


def _positive_median(values: torch.Tensor) -> float:
    positive = values[torch.isfinite(values) & (values > 0)]
    if positive.numel() == 0:
        return 1.0
    return max(float(positive.median().item()), 1e-20)


def build_fisher_mask(
    model: torch.nn.Module,
    retain_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    *,
    top_fraction: float,
    forget_absolute_quantile: float,
    epsilon: float = 1e-12,
    include_bias: bool = False,
    include_batchnorm_affine: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Seleziona pesi forget-specific tramite Fisher ratio e gate assoluto."""
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction deve essere in (0, 1].")
    if not 0.0 <= forget_absolute_quantile < 1.0:
        raise ValueError("forget_absolute_quantile deve essere in [0, 1).")
    eligible = _eligible_parameter_names(
        model,
        include_bias=include_bias,
        include_batchnorm_affine=include_batchnorm_affine,
    )
    missing = eligible - set(retain_fisher) | eligible - set(forget_fisher)
    if missing:
        raise KeyError(f"Fisher mancanti per i parametri: {sorted(missing)}")
    parameter_map = dict(model.named_parameters())
    for name in eligible:
        expected_shape = parameter_map[name].shape
        if (
            retain_fisher[name].shape != expected_shape
            or forget_fisher[name].shape != expected_shape
        ):
            raise ValueError(f"Shape Fisher incompatibile per {name}.")

    retain_values = torch.cat(
        [retain_fisher[name].float().reshape(-1) for name in eligible]
    )
    forget_values = torch.cat(
        [forget_fisher[name].float().reshape(-1) for name in eligible]
    )
    retain_scale = _positive_median(retain_values)
    forget_scale = _positive_median(forget_values)
    normalized_forget = forget_values / forget_scale
    absolute_threshold = float(
        torch.quantile(normalized_forget, forget_absolute_quantile).item()
    )

    scores: dict[str, torch.Tensor] = {}
    valid_locations: list[tuple[str, torch.Tensor]] = []
    valid_scores: list[torch.Tensor] = []
    eligible_count = 0
    for name, parameter in model.named_parameters():
        if name not in eligible:
            continue
        normalized_retain = retain_fisher[name].float() / retain_scale
        normalized_forget_layer = forget_fisher[name].float() / forget_scale
        score = torch.log(normalized_forget_layer + epsilon) - torch.log(
            normalized_retain + epsilon
        )
        gate = (
            torch.isfinite(score)
            & torch.isfinite(normalized_forget_layer)
            & (normalized_forget_layer >= absolute_threshold)
        )
        scores[name] = score
        locations = torch.nonzero(gate.reshape(-1), as_tuple=False).squeeze(1)
        if locations.numel():
            valid_locations.append((name, locations))
            valid_scores.append(score.reshape(-1)[locations])
        eligible_count += parameter.numel()

    candidate_count = int(sum(values.numel() for values in valid_scores))
    if candidate_count == 0:
        raise RuntimeError("Il gate Fisher non ha prodotto parametri candidati.")
    selected_count = min(
        max(1, int(round(eligible_count * top_fraction))), candidate_count
    )
    selected_global = torch.topk(
        torch.cat(valid_scores), k=selected_count, largest=True, sorted=False
    ).indices
    selected_flags = torch.zeros(candidate_count, dtype=torch.bool)
    selected_flags[selected_global] = True

    masks = {
        name: torch.zeros_like(parameter, dtype=torch.bool, device="cpu")
        for name, parameter in model.named_parameters()
    }
    cursor = 0
    for name, locations in valid_locations:
        length = locations.numel()
        chosen = selected_flags[cursor : cursor + length]
        masks[name].view(-1)[locations[chosen]] = True
        cursor += length

    layer_statistics: list[dict[str, Any]] = []
    actual_selected = 0
    for name, parameter in model.named_parameters():
        selected = int(masks[name].sum().item())
        actual_selected += selected
        layer_statistics.append(
            {
                "parameter": name,
                "numel": parameter.numel(),
                "selected": selected,
                "selected_fraction": selected / max(parameter.numel(), 1),
                "eligible": name in eligible,
            }
        )
    return (
        masks,
        scores,
        {
            "retain_scale_median": retain_scale,
            "forget_scale_median": forget_scale,
            "forget_absolute_quantile": forget_absolute_quantile,
            "forget_absolute_threshold_normalized": absolute_threshold,
            "requested_top_fraction": top_fraction,
            "selected": actual_selected,
            "eligible_numel": eligible_count,
            "selected_fraction_of_eligible": actual_selected / max(eligible_count, 1),
            "candidate_count_after_gate": candidate_count,
            "layer_statistics": layer_statistics,
        },
    )


def apply_selective_fisher_dampening(
    model_builder: ModelBuilder,
    original_state: Mapping[str, torch.Tensor],
    retain_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    minimum_factor: float,
    strength: float,
    ratio_power: float,
    epsilon: float = 1e-12,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Attenua soltanto gli elementi selezionati, partendo dallo stato originale."""
    if not 0.0 < minimum_factor <= 1.0:
        raise ValueError("minimum_factor deve essere in (0, 1].")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength deve essere in [0, 1].")
    if ratio_power <= 0.0:
        raise ValueError("ratio_power deve essere positivo.")

    model = model_builder(original_state).to(device)
    model.eval()
    selected_names = [name for name, mask in masks.items() if bool(mask.any())]
    if not selected_names:
        raise ValueError("La maschera Fisher non seleziona alcun parametro.")
    retain_scale = _positive_median(
        torch.cat([retain_fisher[name].float().reshape(-1) for name in selected_names])
    )
    forget_scale = _positive_median(
        torch.cat([forget_fisher[name].float().reshape(-1) for name in selected_names])
    )

    layer_statistics: list[dict[str, Any]] = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            mask_cpu = masks.get(name)
            if mask_cpu is None or not bool(mask_cpu.any()):
                continue
            mask = mask_cpu.to(device)
            normalized_retain = (
                retain_fisher[name].to(device, parameter.dtype) / retain_scale
            )
            normalized_forget = (
                forget_fisher[name].to(device, parameter.dtype) / forget_scale
            )
            raw_factor = (normalized_retain + epsilon) / (normalized_forget + epsilon)
            raw_factor = raw_factor.clamp(max=1.0).pow(ratio_power)
            raw_factor = raw_factor.clamp(min=minimum_factor, max=1.0)
            factor = 1.0 - strength * (1.0 - raw_factor)
            before = parameter.detach().clone()
            parameter[mask] *= factor[mask]
            layer_statistics.append(
                {
                    "parameter": name,
                    "modified": int(mask.sum().item()),
                    "factor_min": float(factor[mask].min().item()),
                    "factor_mean": float(factor[mask].mean().item()),
                    "factor_max": float(factor[mask].max().item()),
                    "delta_l2": float((parameter - before).norm().item()),
                }
            )
    assert_finite_state(model.state_dict())
    model.eval()
    return model, {
        "minimum_factor": minimum_factor,
        "strength": strength,
        "ratio_power": ratio_power,
        "modified_total": int(sum(row["modified"] for row in layer_statistics)),
        "layer_statistics": layer_statistics,
    }


def _freeze_batchnorm_statistics(model: torch.nn.Module) -> None:
    """Lascia trainabili i pesi del modello ma congela i buffer BatchNorm."""
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def _mask_gradients(
    model: torch.nn.Module,
    masks: Mapping[str, torch.Tensor],
    *,
    update_selected: bool,
) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        mask = masks.get(name)
        if mask is None:
            if update_selected:
                parameter.grad.zero_()
            continue
        mask_device = mask.to(parameter.grad.device)
        parameter.grad.mul_(
            (mask_device if update_selected else ~mask_device).to(parameter.grad.dtype)
        )


def _restore_selected_parameters(
    model: torch.nn.Module,
    protected_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
) -> None:
    """Neutralizza anche gli aggiornamenti AdamW dovuti al weight decay."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            mask = masks.get(name)
            if mask is None or not bool(mask.any()):
                continue
            mask_device = mask.to(parameter.device)
            protected = protected_state[name].to(parameter.device, parameter.dtype)
            parameter[mask_device] = protected[mask_device]


def selective_gradient_ascent(
    model: torch.nn.Module,
    *,
    forget_features: np.ndarray,
    forget_targets: np.ndarray,
    retain_features: np.ndarray,
    retain_teacher_logits: np.ndarray,
    masks: Mapping[str, torch.Tensor],
    device: torch.device,
    seed: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
    retain_distillation_weight: float,
    gradient_clip: float,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Aumenta la loss forget aggiornando solo i parametri selezionati."""
    if steps <= 0:
        model.eval()
        return model, pd.DataFrame()
    seed_everything(seed)
    model.train()
    _freeze_batchnorm_statistics(model)
    forget_loader = make_data_loader(
        forget_features,
        forget_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
    )
    retain_loader = make_data_loader(
        retain_features,
        retain_teacher_logits,
        batch_size=batch_size,
        shuffle=True,
        seed=seed + 1,
        device=device,
    )
    forget_iterator = iter(forget_loader)
    retain_iterator = iter(retain_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        try:
            forget_batch, forget_target_batch = next(forget_iterator)
        except StopIteration:
            forget_iterator = iter(forget_loader)
            forget_batch, forget_target_batch = next(forget_iterator)
        try:
            retain_batch, teacher_batch = next(retain_iterator)
        except StopIteration:
            retain_iterator = iter(retain_loader)
            retain_batch, teacher_batch = next(retain_iterator)

        forget_batch = forget_batch.to(device)
        forget_target_batch = forget_target_batch.to(device)
        retain_batch = retain_batch.to(device)
        teacher_batch = teacher_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        forget_loss = F.binary_cross_entropy_with_logits(
            model(forget_batch), forget_target_batch
        )
        retain_distillation = F.mse_loss(model(retain_batch), teacher_batch)
        objective = -forget_loss + retain_distillation_weight * retain_distillation
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError("Obiettivo non finito nel gradient ascent.")
        objective.backward()
        _mask_gradients(model, masks, update_selected=True)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Gradienti non finiti nel gradient ascent.")
        optimizer.step()
        history.append(
            {
                "step": step,
                "forget_bce": float(forget_loss.detach().item()),
                "retain_distillation_mse": float(retain_distillation.detach().item()),
                "objective": float(objective.detach().item()),
                "gradient_norm": float(gradient_norm),
            }
        )
    assert_finite_state(model.state_dict())
    model.eval()
    return model, pd.DataFrame(history)


def _parameter_regularization(
    model: torch.nn.Module,
    original_state: Mapping[str, torch.Tensor],
    protected_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    *,
    selected_weight: float,
) -> torch.Tensor:
    """Calcola una media globale per elemento, indipendente dalla dimensione layer."""
    total = torch.zeros((), device=next(model.parameters()).device)
    weighted_count = 0.0
    for name, parameter in model.named_parameters():
        mask = masks[name].to(parameter.device)
        original = original_state[name].to(parameter.device, parameter.dtype)
        protected = protected_state[name].to(parameter.device, parameter.dtype)
        if bool((~mask).any()):
            total = total + (parameter[~mask] - original[~mask]).square().sum()
            weighted_count += int((~mask).sum().item())
        if selected_weight > 0.0 and bool(mask.any()):
            total = (
                total
                + selected_weight * (parameter[mask] - protected[mask]).square().sum()
            )
            weighted_count += selected_weight * int(mask.sum().item())
    return total / max(weighted_count, 1.0)


def _repair_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    positive_class_weights: torch.Tensor,
    original_state: Mapping[str, torch.Tensor],
    protected_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    supervised_weight: float,
    distillation_weight: float,
    parameter_regularization_weight: float,
    selected_parameter_weight: float,
    gradient_clip: float,
    freeze_selected: bool,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    _freeze_batchnorm_statistics(model)
    totals = {"loss": 0.0, "supervised": 0.0, "distillation": 0.0}
    processed = 0
    for feature_batch, target_batch, teacher_batch in loader:
        feature_batch = feature_batch.to(device)
        target_batch = target_batch.to(device)
        teacher_batch = teacher_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        student_logits = model(feature_batch)
        supervised = F.binary_cross_entropy_with_logits(
            student_logits,
            target_batch,
            pos_weight=positive_class_weights,
        )
        distillation = F.mse_loss(student_logits, teacher_batch)
        parameter_penalty = _parameter_regularization(
            model,
            original_state,
            protected_state,
            masks,
            selected_weight=selected_parameter_weight,
        )
        loss = (
            supervised_weight * supervised
            + distillation_weight * distillation
            + parameter_regularization_weight * parameter_penalty
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Loss non finita durante il repair.")
        loss.backward()
        if freeze_selected:
            _mask_gradients(model, masks, update_selected=False)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), gradient_clip
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("Gradienti non finiti durante il repair.")
        optimizer.step()
        if freeze_selected:
            _restore_selected_parameters(model, protected_state, masks)

        batch_count = len(feature_batch)
        totals["loss"] += float(loss.detach().item()) * batch_count
        totals["supervised"] += float(supervised.detach().item()) * batch_count
        totals["distillation"] += float(distillation.detach().item()) * batch_count
        processed += batch_count
    return {
        "train_loss": totals["loss"] / max(processed, 1),
        "supervised_bce": totals["supervised"] / max(processed, 1),
        "distillation_mse": totals["distillation"] / max(processed, 1),
    }


def repair_with_distillation_fixed(
    model: torch.nn.Module,
    *,
    retain_features: np.ndarray,
    retain_targets: np.ndarray,
    retain_teacher_logits: np.ndarray,
    original_state: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor],
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    positive_class_weights: torch.Tensor,
    supervised_weight: float,
    distillation_weight: float,
    parameter_regularization_weight: float,
    selected_parameter_weight: float,
    gradient_clip: float,
    freeze_selected: bool,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Ripara il modello per epoche fisse senza usare la validation."""
    if len(retain_features) != len(retain_teacher_logits):
        raise ValueError("Teacher logits e retain set non sono allineati.")
    if epochs <= 0:
        model.eval()
        return model, pd.DataFrame()
    seed_everything(seed)
    protected_state = model_state_to_cpu(model)
    loader = make_data_loader(
        retain_features,
        retain_targets,
        retain_teacher_logits,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    class_weights = positive_class_weights.to(device)
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        epoch_metrics = _repair_one_epoch(
            model,
            loader,
            optimizer,
            positive_class_weights=class_weights,
            original_state=original_state,
            protected_state=protected_state,
            masks=masks,
            supervised_weight=supervised_weight,
            distillation_weight=distillation_weight,
            parameter_regularization_weight=parameter_regularization_weight,
            selected_parameter_weight=selected_parameter_weight,
            gradient_clip=gradient_clip,
            freeze_selected=freeze_selected,
            device=device,
        )
        history.append({"epoch": epoch, **epoch_metrics})
    assert_finite_state(model.state_dict())
    model.eval()
    return model, pd.DataFrame(history)


def recalibrate_batchnorm(
    model: torch.nn.Module,
    retain_features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> None:
    """Ricalibra i soli buffer BatchNorm usando esclusivamente il retain set."""
    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not batchnorm_modules:
        model.eval()
        return
    original_momenta = [module.momentum for module in batchnorm_modules]
    model.eval()
    for module in batchnorm_modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()
    loader = make_data_loader(
        retain_features,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
        batchnorm_training=True,
    )
    with torch.inference_mode():
        for (feature_batch,) in loader:
            model(feature_batch.to(device))
    for module, momentum in zip(batchnorm_modules, original_momenta):
        module.momentum = momentum
    model.eval()
    assert_finite_state(model.state_dict())


def precompute_teacher_logits(
    teacher_model: torch.nn.Module,
    retain_features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Precalcola logit teacher mantenendo l'ordine del retain set."""
    return predict_logits(
        teacher_model,
        retain_features,
        device=device,
        batch_size=batch_size,
    ).astype(np.float32, copy=False)


def _build_mask_and_dampened_model(
    config: Mapping[str, Any],
    *,
    model_builder: ModelBuilder,
    original_state: Mapping[str, torch.Tensor],
    retain_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    model_for_mask = model_builder(original_state).to(device).eval()
    masks, _, mask_metadata = build_fisher_mask(
        model_for_mask,
        retain_fisher,
        forget_fisher,
        top_fraction=float(config["top_fraction"]),
        forget_absolute_quantile=float(config["forget_absolute_quantile"]),
        include_bias=bool(config.get("include_bias", False)),
        include_batchnorm_affine=bool(config.get("include_batchnorm_affine", False)),
    )
    del model_for_mask
    model, dampening_metadata = apply_selective_fisher_dampening(
        model_builder,
        original_state,
        retain_fisher,
        forget_fisher,
        masks,
        device=device,
        minimum_factor=float(config["minimum_dampening_factor"]),
        strength=float(config["dampening_strength"]),
        ratio_power=float(config["fisher_ratio_power"]),
    )
    return model, masks, mask_metadata, dampening_metadata


def run_fixed_hybrid_unlearning(
    config: Mapping[str, Any],
    *,
    original_model: torch.nn.Module,
    model_builder: ModelBuilder,
    original_state: Mapping[str, torch.Tensor],
    retain_features: np.ndarray,
    retain_targets: np.ndarray,
    forget_features: np.ndarray,
    forget_targets: np.ndarray,
    device: torch.device,
    seed: int,
) -> HybridUnlearningResult:
    """Esegue il metodo ibrido fisso misurandone tutte le fasi necessarie."""
    retain_sample_size = min(
        int(config["fisher_retain_sample_size"]), len(retain_features)
    )
    forget_sample_size = min(
        int(config["fisher_forget_sample_size"]), len(forget_features)
    )
    if device.type == "cpu" and retain_sample_size + forget_sample_size > 8000:
        warnings.warn(
            "La configurazione Fisher selezionata puo' richiedere diversi minuti su CPU.",
            RuntimeWarning,
            stacklevel=2,
        )

    start = time.perf_counter()
    seed_everything(seed)
    teacher_logits = precompute_teacher_logits(
        original_model,
        retain_features,
        device=device,
        batch_size=int(config["teacher_batch_size"]),
    )
    retain_fisher, retain_fisher_metadata = compute_diagonal_fisher(
        original_model,
        retain_features,
        retain_targets,
        device=device,
        sample_size=retain_sample_size,
        batch_size=int(config["fisher_batch_size"]),
        seed=seed,
        include_bias=bool(config.get("include_bias", False)),
        include_batchnorm_affine=bool(config.get("include_batchnorm_affine", False)),
    )
    forget_fisher, forget_fisher_metadata = compute_diagonal_fisher(
        original_model,
        forget_features,
        forget_targets,
        device=device,
        sample_size=forget_sample_size,
        batch_size=int(config["fisher_batch_size"]),
        seed=seed + 1,
        include_bias=bool(config.get("include_bias", False)),
        include_batchnorm_affine=bool(config.get("include_batchnorm_affine", False)),
    )
    model, masks, mask_metadata, dampening_metadata = _build_mask_and_dampened_model(
        config,
        model_builder=model_builder,
        original_state=original_state,
        retain_fisher=retain_fisher,
        forget_fisher=forget_fisher,
        device=device,
    )
    model, ascent_history = selective_gradient_ascent(
        model,
        forget_features=forget_features,
        forget_targets=forget_targets,
        retain_features=retain_features,
        retain_teacher_logits=teacher_logits,
        masks=masks,
        device=device,
        seed=seed,
        learning_rate=float(config["gradient_ascent_learning_rate"]),
        steps=int(config["gradient_ascent_steps"]),
        batch_size=int(config["gradient_ascent_batch_size"]),
        retain_distillation_weight=float(
            config["gradient_ascent_retain_distillation_weight"]
        ),
        gradient_clip=float(config["gradient_clip"]),
    )
    positive_class_weights = compute_positive_class_weights(
        retain_targets, device=device
    )
    model, repair_history = repair_with_distillation_fixed(
        model,
        retain_features=retain_features,
        retain_targets=retain_targets,
        retain_teacher_logits=teacher_logits,
        original_state=original_state,
        masks=masks,
        device=device,
        seed=seed,
        epochs=int(config["fixed_repair_epochs"]),
        learning_rate=float(config["repair_learning_rate"]),
        weight_decay=float(config["repair_weight_decay"]),
        batch_size=int(config["repair_batch_size"]),
        positive_class_weights=positive_class_weights,
        supervised_weight=float(config["supervised_loss_weight"]),
        distillation_weight=float(config["distillation_weight"]),
        parameter_regularization_weight=float(
            config["parameter_regularization_weight"]
        ),
        selected_parameter_weight=float(config["selected_parameter_weight"]),
        gradient_clip=float(config["gradient_clip"]),
        freeze_selected=bool(config["freeze_selected_during_repair"]),
    )
    if bool(config["recalibrate_batchnorm"]):
        recalibrate_batchnorm(
            model,
            retain_features,
            device=device,
            batch_size=int(config["batchnorm_recalibration_batch_size"]),
        )
    elapsed = time.perf_counter() - start
    state_dict = model_state_to_cpu(model)
    assert_finite_state(state_dict)
    return HybridUnlearningResult(
        model=model,
        state_dict=state_dict,
        execution_time_seconds=float(elapsed),
        metadata={
            "retain_fisher": retain_fisher_metadata,
            "forget_fisher": forget_fisher_metadata,
            "mask": mask_metadata,
            "dampening": dampening_metadata,
        },
        repair_history=repair_history,
        gradient_ascent_history=ascent_history,
    )


def execute_search_candidate(
    config: Mapping[str, Any],
    *,
    model_builder: ModelBuilder,
    original_state: Mapping[str, torch.Tensor],
    retain_fisher: Mapping[str, torch.Tensor],
    forget_fisher: Mapping[str, torch.Tensor],
    retain_features: np.ndarray,
    retain_targets: np.ndarray,
    retain_teacher_logits: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    forget_features: np.ndarray,
    forget_targets: np.ndarray,
    privacy_proxy: ReferencePrivacyProxy,
    baseline_precision_at_10: float,
    retraining_time_seconds: float,
    positive_class_weights: torch.Tensor,
    device: torch.device,
    seed: int,
    shared_method_time_seconds: float,
) -> dict[str, Any]:
    """Valuta un candidato con early stopping, solo nel workflow di ricerca."""
    seed_everything(seed)
    method_start = time.perf_counter()
    model, masks, mask_metadata, dampening_metadata = _build_mask_and_dampened_model(
        config,
        model_builder=model_builder,
        original_state=original_state,
        retain_fisher=retain_fisher,
        forget_fisher=forget_fisher,
        device=device,
    )
    method_elapsed = shared_method_time_seconds + (time.perf_counter() - method_start)
    ascent_start = time.perf_counter()
    model, ascent_history = selective_gradient_ascent(
        model,
        forget_features=forget_features,
        forget_targets=forget_targets,
        retain_features=retain_features,
        retain_teacher_logits=retain_teacher_logits,
        masks=masks,
        device=device,
        seed=seed,
        learning_rate=float(config.get("gradient_ascent_learning_rate", 1e-7)),
        steps=int(config.get("gradient_ascent_steps", 0)),
        batch_size=int(config.get("gradient_ascent_batch_size", 128)),
        retain_distillation_weight=float(
            config.get("gradient_ascent_retain_distillation_weight", 1.0)
        ),
        gradient_clip=float(config["gradient_clip"]),
    )
    method_elapsed += time.perf_counter() - ascent_start

    protected_state = model_state_to_cpu(model)
    loader = make_data_loader(
        retain_features,
        retain_targets,
        retain_teacher_logits,
        batch_size=int(config["repair_batch_size"]),
        shuffle=True,
        seed=seed,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["repair_learning_rate"]),
        weight_decay=float(config["repair_weight_decay"]),
    )

    def candidate_metrics(current_time: float) -> dict[str, float]:
        return evaluate_unlearning_candidate(
            model,
            validation_features=validation_features,
            validation_targets=validation_targets,
            forget_features=forget_features,
            forget_targets=forget_targets,
            device=device,
            privacy_proxy=privacy_proxy,
            baseline_precision_at_10=baseline_precision_at_10,
            retraining_time_seconds=retraining_time_seconds,
            execution_time_seconds=current_time,
        )

    best_metrics = candidate_metrics(method_elapsed)
    best_state = model_state_to_cpu(model)
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = [{"epoch": 0, **best_metrics}]
    utility_floor = baseline_precision_at_10 * float(config["utility_floor_ratio"])

    for epoch in range(1, int(config["repair_max_epochs"]) + 1):
        train_start = time.perf_counter()
        training_metrics = _repair_one_epoch(
            model,
            loader,
            optimizer,
            positive_class_weights=positive_class_weights.to(device),
            original_state=original_state,
            protected_state=protected_state,
            masks=masks,
            supervised_weight=float(config["supervised_loss_weight"]),
            distillation_weight=float(config["distillation_weight"]),
            parameter_regularization_weight=float(
                config["parameter_regularization_weight"]
            ),
            selected_parameter_weight=float(config["selected_parameter_weight"]),
            gradient_clip=float(config["gradient_clip"]),
            freeze_selected=bool(config["freeze_selected_during_repair"]),
            device=device,
        )
        method_elapsed += time.perf_counter() - train_start
        metrics = candidate_metrics(method_elapsed)
        history.append({"epoch": epoch, **training_metrics, **metrics})
        utility_ok = metrics["precision_at_10"] >= utility_floor
        best_utility_ok = best_metrics["precision_at_10"] >= utility_floor
        improves = utility_ok and not best_utility_ok
        if utility_ok == best_utility_ok:
            improves = (
                metrics["local_search_score"]
                > best_metrics["local_search_score"] + 1e-12
            )
        if improves:
            best_metrics = metrics
            best_state = model_state_to_cpu(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["repair_patience"]):
                break

    model.load_state_dict(best_state, strict=True)
    if bool(config["recalibrate_batchnorm"]):
        recalibration_start = time.perf_counter()
        recalibrate_batchnorm(
            model,
            retain_features,
            device=device,
            batch_size=int(config["batchnorm_recalibration_batch_size"]),
        )
        method_elapsed += time.perf_counter() - recalibration_start
        best_metrics = candidate_metrics(method_elapsed)

    best_metrics["execution_time_seconds"] = float(method_elapsed)
    best_metrics["best_epoch"] = int(best_epoch)
    best_metrics["utility_floor_pass"] = bool(
        best_metrics["precision_at_10"] >= utility_floor
    )
    return {
        "state_dict": model_state_to_cpu(model),
        "metrics": best_metrics,
        "config": deepcopy(dict(config)),
        "mask_metadata": mask_metadata,
        "dampening_metadata": dampening_metadata,
        "repair_history": pd.DataFrame(history),
        "gradient_ascent_history": ascent_history,
    }


def select_best_search_result(
    results: Iterable[dict[str, Any]],
    *,
    baseline_precision_at_10: float,
    utility_floor_ratio: float,
) -> dict[str, Any]:
    """Seleziona il candidato migliore rispettando prima il vincolo di utility."""
    result_list = list(results)
    if not result_list:
        raise ValueError("Nessun risultato da selezionare.")
    floor = baseline_precision_at_10 * utility_floor_ratio
    feasible = [
        result
        for result in result_list
        if result["metrics"]["precision_at_10"] >= floor
    ]
    pool = feasible or result_list
    return max(
        pool,
        key=lambda result: (
            result["metrics"]["local_search_score"],
            result["metrics"]["local_privacy_proxy"],
            result["metrics"]["precision_at_10"],
            -result["metrics"]["execution_time_seconds"],
        ),
    )


def progressive_search(
    configurations: Iterable[dict[str, Any]],
    *,
    execute_kwargs: dict[str, Any],
    baseline_precision_at_10: float,
    utility_floor_ratio: float,
    add_gradient_ascent_variants: int,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Prova le configurazioni e aggiunge GA soltanto alle migliori."""
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for index, configuration in enumerate(configurations, start=1):
        result = execute_search_candidate(configuration, **execute_kwargs)
        results.append(result)
        rows.append({"config_index": index, **configuration, **result["metrics"]})
        metrics = result["metrics"]
        print(
            f"[{index}] {configuration['name']} | "
            f"P@10={metrics['precision_at_10']:.6f} | "
            f"BCE={metrics['validation_bce']:.6f} | "
            f"privacy_proxy={metrics['local_privacy_proxy']:.6f} | "
            f"selected={result['mask_metadata']['selected_fraction_of_eligible']:.2%} | "
            f"time={metrics['execution_time_seconds']:.2f}s | "
            f"score={metrics['local_search_score']:.6f}"
        )
        release_memory()

    ranked = sorted(
        results,
        key=lambda item: item["metrics"]["local_search_score"],
        reverse=True,
    )[:add_gradient_ascent_variants]
    for base_result in ranked:
        configuration = deepcopy(base_result["config"])
        configuration.update(
            {
                "name": configuration["name"] + "_ga",
                "gradient_ascent_steps": 4,
                "gradient_ascent_learning_rate": max(
                    float(configuration["repair_learning_rate"]) * 0.1, 1e-7
                ),
            }
        )
        result = execute_search_candidate(configuration, **execute_kwargs)
        results.append(result)
        rows.append(
            {
                "config_index": len(rows) + 1,
                **configuration,
                **result["metrics"],
            }
        )
        metrics = result["metrics"]
        print(
            f"[{len(rows)}] {configuration['name']} | "
            f"P@10={metrics['precision_at_10']:.6f} | "
            f"BCE={metrics['validation_bce']:.6f} | "
            f"privacy_proxy={metrics['local_privacy_proxy']:.6f} | "
            f"selected={result['mask_metadata']['selected_fraction_of_eligible']:.2%} | "
            f"time={metrics['execution_time_seconds']:.2f}s | "
            f"score={metrics['local_search_score']:.6f}"
        )
        release_memory()

    best = select_best_search_result(
        results,
        baseline_precision_at_10=baseline_precision_at_10,
        utility_floor_ratio=utility_floor_ratio,
    )
    comparison = pd.DataFrame(rows).sort_values(
        [
            "utility_floor_pass",
            "local_search_score",
            "local_privacy_proxy",
            "precision_at_10",
        ],
        ascending=[False, False, False, False],
    )
    return best, comparison, results
