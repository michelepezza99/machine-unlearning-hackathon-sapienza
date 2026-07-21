"""Orchestrazione autonoma del metodo finale selezionato."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .data import load_challenge_data, validate_data_model_compatibility
from .metrics import evaluate_model
from .model import (
    build_model,
    build_model_from_artifact,
    load_model_artifact,
    model_state_to_cpu,
)
from .submission import create_submission, validate_submission
from .training import (
    compute_positive_class_weights,
    seed_everything,
    synchronize_device,
    train_fixed_epochs,
    validate_seed,
)
from .unlearning import run_fixed_hybrid_unlearning


SUPPORTED_FINAL_METHODS = {
    "retraining_from_scratch",
    "hybrid_fisher_dampening",
}

RETRAINING_REQUIRED_FIELDS = {
    "optimizer",
    "learning_rate",
    "weight_decay",
    "training_batch_size",
    "fixed_epochs",
}

HYBRID_REQUIRED_FIELDS = {
    "teacher_batch_size",
    "fisher_retain_sample_size",
    "fisher_forget_sample_size",
    "fisher_batch_size",
    "top_fraction",
    "forget_absolute_quantile",
    "minimum_dampening_factor",
    "dampening_strength",
    "fisher_ratio_power",
    "gradient_ascent_steps",
    "gradient_ascent_learning_rate",
    "gradient_ascent_batch_size",
    "gradient_ascent_retain_distillation_weight",
    "repair_learning_rate",
    "repair_weight_decay",
    "repair_batch_size",
    "fixed_repair_epochs",
    "supervised_loss_weight",
    "distillation_weight",
    "parameter_regularization_weight",
    "selected_parameter_weight",
    "gradient_clip",
    "freeze_selected_during_repair",
    "recalibrate_batchnorm",
    "batchnorm_recalibration_batch_size",
}

FINAL_DIAGNOSTIC_FILENAMES = {
    "final_training_history.csv",
    "final_gradient_ascent_history.csv",
    "final_config_used.json",
    "final_metrics.json",
    "method_metadata.json",
}

TIMING_POLICY_BY_METHOD = {
    "retraining_from_scratch": (
        "Esclude caricamento di dati e artifact, valutazione post-hoc, "
        "serializzazione e validazione della submission. Include seed, costruzione "
        "del modello, pesi di classe, optimizer e tutte le epoche fisse."
    ),
    "hybrid_fisher_dampening": (
        "Esclude caricamento di dati e artifact, ricostruzione del modello originale, "
        "valutazione post-hoc, serializzazione e validazione della submission. Include "
        "seed, teacher logits, Fisher retain/forget, maschera e dampening, gradient "
        "ascent opzionale, pesi di classe, repair e ricalibrazione BatchNorm opzionale."
    ),
}


@dataclass
class FinalRunSummary:
    """Riepilogo essenziale restituito dal workflow finale."""

    method: str
    device: str
    execution_time_seconds: float
    submission_dir: Path
    metrics: dict[str, float]
    validation: dict[str, Any]


def _require_fields(
    config: Mapping[str, Any], required: set[str], *, context: str
) -> None:
    missing = required - set(config)
    if missing:
        raise KeyError(f"Configurazione {context} incompleta: {sorted(missing)}")


def _require_integer(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} deve essere un intero.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} deve essere almeno {minimum}.")
    return value


def _require_number(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} deve essere un numero.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} deve essere finito.")
    if minimum is not None:
        below_minimum = number < minimum if minimum_inclusive else number <= minimum
        if below_minimum:
            comparator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{key} deve essere {comparator} {minimum}.")
    if maximum is not None:
        above_maximum = number > maximum if maximum_inclusive else number >= maximum
        if above_maximum:
            comparator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{key} deve essere {comparator} {maximum}.")
    return number


def _require_boolean(config: Mapping[str, Any], key: str) -> bool:
    value = config[key]
    if not isinstance(value, bool):
        raise TypeError(f"{key} deve essere booleano.")
    return value


def validate_final_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Valida una configurazione finale prima di caricare dati o modelli."""
    if not isinstance(config, Mapping):
        raise TypeError("La configurazione JSON deve contenere un oggetto.")
    required_common = {"schema_version", "method", "seed", "validation_fraction"}
    _require_fields(config, required_common, context="finale")

    if _require_integer(config, "schema_version") != 1:
        raise ValueError("schema_version supportata: 1.")
    method = config["method"]
    if not isinstance(method, str) or method not in SUPPORTED_FINAL_METHODS:
        raise ValueError(
            f"Metodo finale non supportato: {method!r}. "
            f"Valori ammessi: {sorted(SUPPORTED_FINAL_METHODS)}."
        )
    validate_seed(_require_integer(config, "seed"))
    _require_number(
        config,
        "validation_fraction",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    if "evaluation_batch_size" in config:
        _require_integer(config, "evaluation_batch_size", minimum=1)

    if method == "retraining_from_scratch":
        _require_fields(config, RETRAINING_REQUIRED_FIELDS, context="retraining")
        optimizer = config["optimizer"]
        if not isinstance(optimizer, str) or optimizer.lower() not in {
            "adam",
            "adamw",
            "sgd",
        }:
            raise ValueError("optimizer deve essere uno tra adam, adamw e sgd.")
        _require_number(
            config, "learning_rate", minimum=0.0, minimum_inclusive=False
        )
        _require_number(config, "weight_decay", minimum=0.0)
        _require_integer(config, "training_batch_size", minimum=1)
        _require_integer(config, "fixed_epochs", minimum=1)
        if "momentum" in config:
            _require_number(config, "momentum", minimum=0.0)
        if config.get("gradient_clip") is not None:
            _require_number(
                config, "gradient_clip", minimum=0.0, minimum_inclusive=False
            )
    else:
        _require_fields(config, HYBRID_REQUIRED_FIELDS, context="ibrida")
        for key in (
            "teacher_batch_size",
            "fisher_retain_sample_size",
            "fisher_forget_sample_size",
            "fisher_batch_size",
            "gradient_ascent_batch_size",
            "repair_batch_size",
            "batchnorm_recalibration_batch_size",
        ):
            _require_integer(config, key, minimum=1)
        _require_integer(config, "gradient_ascent_steps", minimum=0)
        _require_integer(config, "fixed_repair_epochs", minimum=0)
        _require_number(
            config,
            "top_fraction",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _require_number(
            config,
            "forget_absolute_quantile",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _require_number(
            config,
            "minimum_dampening_factor",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _require_number(
            config, "dampening_strength", minimum=0.0, maximum=1.0
        )
        _require_number(
            config, "fisher_ratio_power", minimum=0.0, minimum_inclusive=False
        )
        _require_number(
            config,
            "gradient_ascent_learning_rate",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _require_number(
            config, "gradient_ascent_retain_distillation_weight", minimum=0.0
        )
        _require_number(
            config, "repair_learning_rate", minimum=0.0, minimum_inclusive=False
        )
        _require_number(config, "repair_weight_decay", minimum=0.0)
        for key in (
            "supervised_loss_weight",
            "distillation_weight",
            "parameter_regularization_weight",
            "selected_parameter_weight",
        ):
            _require_number(config, key, minimum=0.0)
        _require_number(
            config, "gradient_clip", minimum=0.0, minimum_inclusive=False
        )
        _require_boolean(config, "freeze_selected_during_repair")
        _require_boolean(config, "recalibrate_batchnorm")
        for optional_boolean in ("include_bias", "include_batchnorm_affine"):
            if optional_boolean in config:
                _require_boolean(config, optional_boolean)

    for optional_text in ("name", "selection_note"):
        if optional_text in config and not isinstance(config[optional_text], str):
            raise TypeError(f"{optional_text} deve essere una stringa.")
    return dict(config)


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Carica e valida una configurazione finale JSON."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configurazione non trovata: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return validate_final_config(config)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"Tipo non serializzabile: {type(value)!r}")


def write_json(path: str | Path, payload: Any) -> Path:
    """Scrive diagnostica JSON leggibile fuori dalla submission."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return destination


def _clear_owned_diagnostics(output_path: Path) -> None:
    """Rimuove solo i file prodotti da questo workflow in esecuzioni precedenti."""
    diagnostic_paths = [
        output_path / filename for filename in sorted(FINAL_DIAGNOSTIC_FILENAMES)
    ]
    non_file_paths = [
        path
        for path in diagnostic_paths
        if path.exists() and not path.is_file() and not path.is_symlink()
    ]
    if non_file_paths:
        raise RuntimeError(
            "I seguenti percorsi diagnostici posseduti non sono file e non vengono "
            f"rimossi automaticamente: {non_file_paths}."
        )
    for diagnostic_path in diagnostic_paths:
        if diagnostic_path.is_file() or diagnostic_path.is_symlink():
            diagnostic_path.unlink()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("E' stato richiesto CUDA, ma non e' disponibile.")
    return device


def _run_fixed_retraining(
    config: Mapping[str, Any],
    *,
    architecture: Mapping[str, Any],
    retain_features: np.ndarray,
    retain_targets: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[
    torch.nn.Module, dict[str, torch.Tensor], float, pd.DataFrame, dict[str, Any]
]:
    """Riesegue il retraining scelto senza early stopping o validation."""
    synchronize_device(device)
    start = time.perf_counter()
    seed_everything(seed)
    model = build_model(architecture, device=device)
    class_weights = compute_positive_class_weights(retain_targets, device=device)
    history = train_fixed_epochs(
        model,
        retain_features,
        retain_targets,
        device=device,
        seed=seed,
        epochs=int(config["fixed_epochs"]),
        batch_size=int(config["training_batch_size"]),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        optimizer_name=str(config["optimizer"]),
        positive_class_weights=class_weights,
        momentum=float(config.get("momentum", 0.0)),
        gradient_clip=(
            float(config["gradient_clip"])
            if config.get("gradient_clip") is not None
            else None
        ),
    )
    synchronize_device(device)
    elapsed = time.perf_counter() - start
    return (
        model,
        model_state_to_cpu(model),
        float(elapsed),
        history,
        {"fixed_epochs": int(config["fixed_epochs"])},
    )


def run_final_workflow(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    submission_dir: str | Path,
    config_path: str | Path,
    seed_override: int | None = None,
    device_name: str = "auto",
) -> FinalRunSummary:
    """Esegue da processo pulito il solo metodo fissato nella configurazione.

    Il timer esclude caricamento dati, validazione, metriche post-hoc e scrittura
    dei file. Include invece ogni operazione necessaria al metodo selezionato.
    """
    config = load_json_config(config_path)
    seed = validate_seed(seed_override if seed_override is not None else config["seed"])
    config = deepcopy(config)
    config["seed"] = seed
    device = _resolve_device(device_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _clear_owned_diagnostics(output_path)

    print("[1/5] Caricamento e verifica dei dati")
    data = load_challenge_data(
        data_dir,
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    original_payload = load_model_artifact(Path(data_dir) / "model_artifact")
    validate_data_model_compatibility(
        data,
        original_payload["architecture"],
        feature_columns=original_payload.get("feature_columns"),
        target_columns=original_payload.get("target_columns"),
    )

    print(f"[2/5] Esecuzione metodo fisso: {config['method']}")
    method = str(config["method"])
    if method == "retraining_from_scratch":
        model, final_state, elapsed, history, method_metadata = _run_fixed_retraining(
            config,
            architecture=original_payload["architecture"],
            retain_features=data.x_retain_train,
            retain_targets=data.y_retain_train,
            device=device,
            seed=seed,
        )
        gradient_ascent_history = pd.DataFrame()
    elif method == "hybrid_fisher_dampening":
        original_model = build_model_from_artifact(original_payload, device=device)
        original_state = model_state_to_cpu(original_model)

        def model_builder(
            state_dict: Mapping[str, torch.Tensor] | None,
        ) -> torch.nn.Module:
            return build_model(
                original_payload["architecture"],
                state_dict=state_dict,
                device=device,
            )

        result = run_fixed_hybrid_unlearning(
            config,
            original_model=original_model,
            model_builder=model_builder,
            original_state=original_state,
            retain_features=data.x_retain_train,
            retain_targets=data.y_retain_train,
            forget_features=data.x_forget,
            forget_targets=data.y_forget,
            device=device,
            seed=seed,
        )
        model = result.model
        final_state = result.state_dict
        elapsed = result.execution_time_seconds
        history = result.repair_history
        gradient_ascent_history = result.gradient_ascent_history
        method_metadata = result.metadata
    else:
        raise ValueError(f"Metodo finale non supportato: {method!r}.")

    method_metadata = {
        **method_metadata,
        "measured_execution_time_seconds": float(elapsed),
        "timing_policy": TIMING_POLICY_BY_METHOD[method],
    }

    print("[3/5] Valutazione post-hoc fuori dal timer")
    validation_metrics = evaluate_model(
        model,
        data.x_validation,
        data.y_validation,
        device=device,
        batch_size=int(config.get("evaluation_batch_size", 2048)),
    )
    forget_metrics = evaluate_model(
        model,
        data.x_forget,
        data.y_forget,
        device=device,
        batch_size=int(config.get("evaluation_batch_size", 2048)),
    )
    final_metrics = {
        "validation_precision_at_10": float(validation_metrics["precision_at_10"]),
        "validation_bce": float(validation_metrics["bce_from_logits"]),
        "forget_bce": float(forget_metrics["bce_from_logits"]),
    }

    print("[4/5] Creazione dei tre file di submission")
    create_submission(
        submission_dir=submission_dir,
        final_state_dict=final_state,
        execution_time_seconds=elapsed,
        validation_ids=data.validation_ids,
        original_payload=original_payload,
        final_config=config,
        final_metrics=final_metrics,
        method_metadata=method_metadata,
        feature_columns=data.schema.feature_columns,
        target_columns=data.schema.target_columns,
        seed=seed,
        validation_fraction=float(config["validation_fraction"]),
    )
    if not history.empty:
        history.to_csv(output_path / "final_training_history.csv", index=False)
    if not gradient_ascent_history.empty:
        gradient_ascent_history.to_csv(
            output_path / "final_gradient_ascent_history.csv", index=False
        )
    write_json(output_path / "final_config_used.json", config)
    write_json(output_path / "final_metrics.json", final_metrics)
    write_json(output_path / "method_metadata.json", method_metadata)

    print("[5/5] Validazione indipendente della submission")
    validation = validate_submission(submission_dir, data_dir=data_dir)
    return FinalRunSummary(
        method=method,
        device=str(device),
        execution_time_seconds=elapsed,
        submission_dir=Path(submission_dir),
        metrics=final_metrics,
        validation=validation,
    )
