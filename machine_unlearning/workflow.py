"""Orchestrazione autonoma del metodo finale selezionato."""

from __future__ import annotations

import json
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
    train_fixed_epochs,
)
from .unlearning import run_fixed_hybrid_unlearning


@dataclass
class FinalRunSummary:
    """Riepilogo essenziale restituito dal workflow finale."""

    method: str
    device: str
    execution_time_seconds: float
    submission_dir: Path
    metrics: dict[str, float]
    validation: dict[str, Any]


def load_json_config(path: str | Path) -> dict[str, Any]:
    """Carica una configurazione JSON e verifica i campi comuni obbligatori."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configurazione non trovata: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"method", "seed", "validation_fraction"}
    missing = required - set(config)
    if missing:
        raise KeyError(f"Configurazione incompleta: {sorted(missing)}")
    return config


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
    seed = int(seed_override if seed_override is not None else config["seed"])
    config = deepcopy(config)
    config["seed"] = seed
    device = _resolve_device(device_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("[1/5] Caricamento e verifica dei dati")
    data = load_challenge_data(
        data_dir,
        validation_fraction=float(config["validation_fraction"]),
        seed=seed,
    )
    original_payload = load_model_artifact(Path(data_dir) / "model_artifact")
    validate_data_model_compatibility(data, original_payload["architecture"])
    if "feature_columns" in original_payload and list(
        original_payload["feature_columns"]
    ) != list(data.schema.feature_columns):
        raise ValueError("Ordine delle feature diverso da quello dell'artifact.")
    if "target_columns" in original_payload and list(
        original_payload["target_columns"]
    ) != list(data.schema.target_columns):
        raise ValueError("Ordine delle target diverso da quello dell'artifact.")

    original_model = build_model_from_artifact(original_payload, device=device)
    original_state = model_state_to_cpu(original_model)
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
        "timing_policy": (
            "Include tutte le operazioni richieste dal metodo dopo il caricamento "
            "di dati e artifact; esclude valutazione post-hoc, serializzazione e "
            "validazione della submission."
        ),
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
