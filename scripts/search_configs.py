"""Esegue baseline, ricerca ibrida e proposta sicura della configurazione finale."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.data import (  # noqa: E402
    ChallengeData,
    load_challenge_data,
    save_validation_ids,
    validate_data_model_compatibility,
)
from machine_unlearning.metrics import (  # noqa: E402
    ReferencePrivacyProxy,
    evaluate_model,
    evaluate_unlearning_candidate,
    fit_reference_privacy_proxy,
)
from machine_unlearning.model import (  # noqa: E402
    build_model,
    build_model_from_artifact,
    load_model_artifact,
    model_state_to_cpu,
)
from machine_unlearning.search import (  # noqa: E402
    build_effective_search_config,
    merge_candidate_configs,
    summarize_multi_seed_results,
)
from machine_unlearning.training import (  # noqa: E402
    compute_positive_class_weights,
    seed_everything,
    synchronize_device,
    train_fixed_epochs,
    train_with_early_stopping,
    validate_seed,
)
from machine_unlearning.unlearning import (  # noqa: E402
    compute_diagonal_fisher,
    precompute_teacher_logits,
    progressive_search,
    release_memory,
    select_best_search_result,
)
from machine_unlearning.workflow import (  # noqa: E402
    validate_final_config,
    write_json,
)


DEFAULT_OUTPUT_DIR = Path("outputs/search")
DEFAULT_PROPOSED_CONFIG = DEFAULT_OUTPUT_DIR / "proposed_final_config.json"
CANONICAL_FINAL_CONFIG = REPOSITORY_ROOT / "configs/final_config.json"

OWNED_RUN_FILES = (
    "validation_ids.csv",
    "retraining_history.csv",
    "search_comparison.csv",
    "finalists.csv",
    "search_metadata.json",
    "best_candidate_summary.json",
    "best_hybrid_repair_history.csv",
    "best_hybrid_gradient_ascent_history.csv",
)

HYBRID_RUNTIME_FIELDS = (
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
    "supervised_loss_weight",
    "distillation_weight",
    "parameter_regularization_weight",
    "selected_parameter_weight",
    "gradient_clip",
    "freeze_selected_during_repair",
    "recalibrate_batchnorm",
    "batchnorm_recalibration_batch_size",
)


@dataclass(frozen=True)
class _BaselinePhase:
    """Dati, artifact e baseline originale condivisi dalle fasi successive."""

    data: ChallengeData
    artifact: dict[str, Any]
    original_model: torch.nn.Module
    original_state: dict[str, torch.Tensor]
    original_forget_logits: np.ndarray
    precision_at_10: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class _RetrainingPhase:
    """Riferimento retrained e misure necessarie al confronto dei candidati."""

    positive_class_weights: torch.Tensor
    positive_class_weight_time: float
    best_epoch: int
    selection_time: float
    fixed_time: float
    privacy_proxy: ReferencePrivacyProxy
    utility_floor: float
    metrics: dict[str, Any]
    search_result: dict[str, Any]


@dataclass(frozen=True)
class _FisherPhase:
    """Teacher e Fisher condivise, calcolate una sola volta per seed."""

    teacher_logits: np.ndarray
    retain_fisher: dict[str, torch.Tensor]
    forget_fisher: dict[str, torch.Tensor]
    retain_metadata: dict[str, Any]
    forget_metadata: dict[str, Any]
    core_time: float
    method_time: float


def build_parser() -> argparse.ArgumentParser:
    """Definisce una CLI esplicita, separata dal workflow finale."""
    parser = argparse.ArgumentParser(
        description="Cerca configurazioni di unlearning senza promuoverle implicitamente.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory dei dati originali della challenge.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/search_configs.json"),
        help="Configurazione JSON della ricerca.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory delle evidenze sperimentali.",
    )
    parser.add_argument(
        "--proposed-config",
        type=Path,
        default=DEFAULT_PROPOSED_CONFIG,
        help="Destinazione della proposta; non promuove la configurazione canonica.",
    )
    parser.add_argument(
        "--device", default="auto", help="Device PyTorch: auto, cpu, cuda o cuda:N."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Usa pochi epoch/campioni/candidati, disabilita le varianti GA e, "
            "con i path predefiniti, scrive in outputs/search/quick."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help=(
            "Limita i candidati ibridi, ma non riduce retraining/Fisher condivise; "
            "usa --quick per ridurre anche quelle fasi."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed opzionali per una valutazione di stabilita' multi-seed.",
    )
    parser.add_argument(
        "--allow-canonical-overwrite",
        action="store_true",
        help="Consente esplicitamente di scrivere configs/final_config.json.",
    )
    return parser


def _load_search_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configurazione di ricerca non trovata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("La configurazione di ricerca deve contenere un oggetto JSON.")
    return payload


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    return device


def _flag_was_provided(arguments: list[str], name: str) -> bool:
    return any(value == name or value.startswith(name + "=") for value in arguments)


def _resolve_output_paths(
    arguments: argparse.Namespace, raw_arguments: list[str]
) -> tuple[Path, Path]:
    output_explicit = _flag_was_provided(raw_arguments, "--output-dir")
    proposed_explicit = _flag_was_provided(raw_arguments, "--proposed-config")
    output_dir = arguments.output_dir
    if arguments.quick and not output_explicit:
        output_dir = DEFAULT_OUTPUT_DIR / "quick"
    proposed_config = arguments.proposed_config
    if not proposed_explicit:
        proposed_config = output_dir / "proposed_final_config.json"
    return output_dir, proposed_config


def _assert_safe_proposed_path(path: Path, *, allow_canonical: bool) -> None:
    if path.resolve() == CANONICAL_FINAL_CONFIG.resolve() and not allow_canonical:
        raise ValueError(
            "Scrittura di configs/final_config.json rifiutata: usa "
            "--allow-canonical-overwrite soltanto dopo aver verificato la proposta."
        )


def _clear_owned_run_files(output_dir: Path, proposed_config: Path) -> None:
    """Remove only files owned by this script so stale evidence cannot survive."""
    for name in OWNED_RUN_FILES:
        destination = output_dir / name
        if destination.is_file():
            destination.unlink()
    if (
        proposed_config.name == "proposed_final_config.json"
        and proposed_config.parent.resolve() == output_dir.resolve()
        and proposed_config.is_file()
    ):
        proposed_config.unlink()


def _clear_multi_seed_files(output_dir: Path) -> None:
    for name in ("multi_seed_results.csv", "multi_seed_summary.csv", "seed_failures.csv"):
        destination = output_dir / name
        if destination.is_file():
            destination.unlink()


def _print_startup_summary(
    *,
    arguments: argparse.Namespace,
    output_dir: Path,
    proposed_config: Path,
    effective_config: Mapping[str, Any],
    seeds: list[int],
    device: torch.device,
) -> None:
    fisher = effective_config["fisher"]
    print("Configurazione effettiva della ricerca")
    print(f"  modalita': {'quick' if arguments.quick else 'full'}")
    print(f"  device: {device}")
    print(f"  seed: {seeds}")
    print(f"  data: {arguments.data_dir}")
    print(f"  output: {output_dir}")
    print(f"  proposta: {proposed_config}")
    print(f"  candidati base: {len(effective_config['candidates'])}")
    print(
        "  campioni Fisher: "
        f"retain={fisher['fisher_retain_sample_size']}, "
        f"forget={fisher['fisher_forget_sample_size']}"
    )
    print(
        "  overwrite config canonica: "
        f"{'ABILITATO' if arguments.allow_canonical_overwrite else 'no'}"
    )


def _build_retraining_final_config(
    search_config: Mapping[str, Any], *, seed: int, best_epoch: int
) -> dict[str, Any]:
    if best_epoch < 1:
        raise ValueError("Il retraining finale richiede almeno un'epoca addestrata.")
    retraining = search_config["retraining"]
    config = {
        "schema_version": 1,
        "name": f"retraining_from_scratch_fixed_{best_epoch}_epochs",
        "method": "retraining_from_scratch",
        "seed": seed,
        "validation_fraction": float(search_config["validation_fraction"]),
        "optimizer": retraining["optimizer"],
        "learning_rate": retraining["learning_rate"],
        "weight_decay": retraining["weight_decay"],
        "training_batch_size": retraining["training_batch_size"],
        "fixed_epochs": int(best_epoch),
        "gradient_clip": None,
        "evaluation_batch_size": int(search_config["evaluation_batch_size"]),
        "selection_note": (
            "Proposta sperimentale locale; verificare i file di evidenza. La proxy "
            "locale non equivale alla MIA ufficiale nascosta."
        ),
        "selection_status": "provisional_search_proposal",
    }
    if "momentum" in retraining:
        config["momentum"] = retraining["momentum"]
    return validate_final_config(config)


def _build_hybrid_final_config(
    search_config: Mapping[str, Any],
    best_hybrid: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    candidate = best_hybrid["config"]
    fisher = search_config["fisher"]
    config: dict[str, Any] = {
        "schema_version": 1,
        "name": str(candidate["name"]),
        "method": "hybrid_fisher_dampening",
        "seed": seed,
        "validation_fraction": float(search_config["validation_fraction"]),
        "evaluation_batch_size": int(search_config["evaluation_batch_size"]),
        **deepcopy(dict(fisher)),
        **{key: deepcopy(candidate[key]) for key in HYBRID_RUNTIME_FIELDS},
        "fixed_repair_epochs": int(best_hybrid["metrics"]["best_epoch"]),
        "selection_note": (
            "Proposta sperimentale locale; verificare i file di evidenza. La proxy "
            "locale non equivale alla MIA ufficiale nascosta."
        ),
        "selection_status": "provisional_search_proposal",
    }
    return validate_final_config(config)


def _result_summary(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    configuration = dict(result["config"])
    return {
        "status": result.get("status", "success"),
        "valid": bool(result.get("valid", True)),
        "effective_configuration": configuration,
        "selected_epoch": int(result["metrics"].get("best_epoch", -1)),
        "metrics": dict(result["metrics"]),
        "selected_parameter_fraction": result.get("mask_metadata", {}).get(
            "selected_fraction_of_eligible"
        ),
        "mask_metadata": result.get("mask_metadata", {}),
        "dampening_metadata": result.get("dampening_metadata", {}),
        "gradient_ascent_used": int(configuration.get("gradient_ascent_steps", 0))
        > 0,
        "batchnorm_recalibration_used": bool(
            configuration.get("recalibrate_batchnorm", False)
        ),
    }


def _write_best_histories(
    output_dir: Path, best_hybrid: Mapping[str, Any] | None
) -> None:
    repair_columns = [
        "epoch",
        "train_loss",
        "supervised_bce",
        "distillation_mse",
        "precision_at_10",
        "validation_bce",
        "forget_bce",
        "local_privacy_proxy",
        "execution_time_seconds",
        "local_search_score",
    ]
    ascent_columns = [
        "step",
        "forget_bce",
        "retain_distillation_mse",
        "objective",
        "gradient_norm",
    ]
    repair = (
        best_hybrid["repair_history"]
        if best_hybrid is not None
        else pd.DataFrame(columns=repair_columns)
    )
    ascent = (
        best_hybrid["gradient_ascent_history"]
        if best_hybrid is not None
        else pd.DataFrame(columns=ascent_columns)
    )
    if repair.empty and not list(repair.columns):
        repair = pd.DataFrame(columns=repair_columns)
    if ascent.empty and not list(ascent.columns):
        ascent = pd.DataFrame(columns=ascent_columns)
    repair.to_csv(output_dir / "best_hybrid_repair_history.csv", index=False)
    ascent.to_csv(output_dir / "best_hybrid_gradient_ascent_history.csv", index=False)


def _finalist_row(
    result: Mapping[str, Any], *, seed: int, selected: bool
) -> dict[str, Any]:
    configuration = result["config"]
    return {
        "seed": seed,
        "method": result["method"],
        "configuration": configuration["name"],
        "status": result.get("status", "success"),
        "valid": bool(result.get("valid", True)),
        "selected": selected,
        "selected_parameter_fraction": result.get("mask_metadata", {}).get(
            "selected_fraction_of_eligible"
        ),
        "gradient_ascent_used": int(configuration.get("gradient_ascent_steps", 0))
        > 0,
        "batchnorm_recalibration_used": bool(
            configuration.get("recalibrate_batchnorm", False)
        ),
        **dict(result["metrics"]),
    }


def _load_baseline_phase(
    *,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    device: torch.device,
    evaluation_batch_size: int,
    validation_fraction: float,
) -> _BaselinePhase:
    """Carica gli input e misura una sola volta la baseline originale."""
    print(f"[seed {seed} | 1/7] Caricamento dati e modello originale")
    seed_everything(seed)
    data = load_challenge_data(
        data_dir,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    save_validation_ids(data, output_dir / "validation_ids.csv")
    artifact = load_model_artifact(data_dir / "model_artifact")
    validate_data_model_compatibility(
        data,
        artifact["architecture"],
        feature_columns=artifact.get("feature_columns"),
        target_columns=artifact.get("target_columns"),
    )
    original_model = build_model_from_artifact(artifact, device=device)
    original_state = model_state_to_cpu(original_model)
    original_validation = evaluate_model(
        original_model,
        data.x_validation,
        data.y_validation,
        device=device,
        batch_size=evaluation_batch_size,
    )
    original_forget = evaluate_model(
        original_model,
        data.x_forget,
        data.y_forget,
        device=device,
        batch_size=evaluation_batch_size,
    )
    precision_at_10 = float(original_validation["precision_at_10"])
    return _BaselinePhase(
        data=data,
        artifact=artifact,
        original_model=original_model,
        original_state=original_state,
        original_forget_logits=original_forget["logits"],
        precision_at_10=precision_at_10,
        metrics={
            "validation_precision_at_10": precision_at_10,
            "validation_bce": float(original_validation["bce_from_logits"]),
            "forget_bce": float(original_forget["bce_from_logits"]),
        },
    )


def _run_retraining_phase(
    seeded_config: Mapping[str, Any],
    *,
    baseline: _BaselinePhase,
    output_dir: Path,
    device: torch.device,
    seed: int,
    evaluation_batch_size: int,
) -> _RetrainingPhase:
    """Seleziona l'epoca retrained e ne riesegue il metodo fisso cronometrato."""
    data = baseline.data
    retraining_config = seeded_config["retraining"]
    print(f"[seed {seed} | 2/7] Selezione epoca e replay fisso del retraining")
    seed_everything(seed)
    selection_model = build_model(baseline.artifact["architecture"], device=device)
    synchronize_device(device)
    weight_start = time.perf_counter()
    positive_class_weights = compute_positive_class_weights(
        data.y_retain_train, device=device
    )
    synchronize_device(device)
    positive_class_weight_time = time.perf_counter() - weight_start
    retraining_result = train_with_early_stopping(
        selection_model,
        data.x_retain_train,
        data.y_retain_train,
        data.x_validation,
        data.y_validation,
        device=device,
        seed=seed,
        max_epochs=int(retraining_config["max_epochs"]),
        patience=int(retraining_config["patience"]),
        batch_size=int(retraining_config["training_batch_size"]),
        evaluation_batch_size=evaluation_batch_size,
        learning_rate=float(retraining_config["learning_rate"]),
        weight_decay=float(retraining_config["weight_decay"]),
        optimizer_name=str(retraining_config["optimizer"]),
        positive_class_weights=positive_class_weights,
        momentum=float(retraining_config.get("momentum", 0.0)),
    )
    retraining_result.history.to_csv(
        output_dir / "retraining_history.csv", index=False
    )
    best_epoch = int(retraining_result.best_epoch)
    if best_epoch < 1:
        raise RuntimeError(
            "La selezione del retraining non ha prodotto un checkpoint addestrato "
            "(best_epoch < 1); il modello casuale non puo' essere finalista."
        )
    selection_time = float(retraining_result.elapsed_seconds)
    del retraining_result, selection_model
    release_memory()

    # Replay the exact fixed method so search-only early stopping is not timed.
    synchronize_device(device)
    retraining_fixed_start = time.perf_counter()
    seed_everything(seed)
    retrained_model = build_model(baseline.artifact["architecture"], device=device)
    replay_class_weights = compute_positive_class_weights(
        data.y_retain_train, device=device
    )
    train_fixed_epochs(
        retrained_model,
        data.x_retain_train,
        data.y_retain_train,
        device=device,
        seed=seed,
        epochs=best_epoch,
        batch_size=int(retraining_config["training_batch_size"]),
        learning_rate=float(retraining_config["learning_rate"]),
        weight_decay=float(retraining_config["weight_decay"]),
        optimizer_name=str(retraining_config["optimizer"]),
        positive_class_weights=replay_class_weights,
        momentum=float(retraining_config.get("momentum", 0.0)),
    )
    synchronize_device(device)
    fixed_time = time.perf_counter() - retraining_fixed_start
    retrained_forget = evaluate_model(
        retrained_model,
        data.x_forget,
        data.y_forget,
        device=device,
        batch_size=evaluation_batch_size,
    )

    print(f"[seed {seed} | 3/7] Proxy locale retrained-reference out-of-fold")
    privacy_proxy = fit_reference_privacy_proxy(
        baseline.original_forget_logits,
        retrained_forget["logits"],
        data.y_forget,
        seed=seed,
    )
    retraining_metrics = evaluate_unlearning_candidate(
        retrained_model,
        validation_features=data.x_validation,
        validation_targets=data.y_validation,
        forget_features=data.x_forget,
        forget_targets=data.y_forget,
        device=device,
        privacy_proxy=privacy_proxy,
        baseline_precision_at_10=baseline.precision_at_10,
        retraining_time_seconds=fixed_time,
        execution_time_seconds=fixed_time,
        batch_size=evaluation_batch_size,
    )
    utility_floor = baseline.precision_at_10 * float(
        seeded_config["utility_floor_ratio"]
    )
    retraining_metrics.update(
        {
            "execution_time_seconds": float(fixed_time),
            "best_epoch": best_epoch,
            "utility_floor_pass": bool(
                retraining_metrics["precision_at_10"] >= utility_floor
            ),
        }
    )
    search_result: dict[str, Any] = {
        "method": "retraining_from_scratch",
        "config": {
            "name": "retraining_reference",
            **deepcopy(dict(retraining_config)),
            "fixed_epochs": best_epoch,
        },
        "config_index": 10**9,
        "status": "success",
        "valid": True,
        "metrics": retraining_metrics,
        "mask_metadata": {},
    }
    del retrained_model, replay_class_weights
    release_memory()
    return _RetrainingPhase(
        positive_class_weights=positive_class_weights,
        positive_class_weight_time=float(positive_class_weight_time),
        best_epoch=best_epoch,
        selection_time=selection_time,
        fixed_time=float(fixed_time),
        privacy_proxy=privacy_proxy,
        utility_floor=utility_floor,
        metrics=retraining_metrics,
        search_result=search_result,
    )


def _compute_fisher_phase(
    seeded_config: Mapping[str, Any],
    *,
    baseline: _BaselinePhase,
    positive_class_weight_time: float,
    device: torch.device,
    seed: int,
) -> _FisherPhase:
    """Precalcola teacher e Fisher condivise includendole nel tempo del metodo."""
    data = baseline.data
    fisher_config = seeded_config["fisher"]
    print(f"[seed {seed} | 4/7] Teacher logits e Fisher condivise")
    synchronize_device(device)
    shared_start = time.perf_counter()
    teacher_logits = precompute_teacher_logits(
        baseline.original_model,
        data.x_retain_train,
        device=device,
        batch_size=int(fisher_config["teacher_batch_size"]),
    )
    retain_fisher, retain_metadata = compute_diagonal_fisher(
        baseline.original_model,
        data.x_retain_train,
        data.y_retain_train,
        device=device,
        sample_size=min(
            int(fisher_config["fisher_retain_sample_size"]),
            len(data.x_retain_train),
        ),
        batch_size=int(fisher_config["fisher_batch_size"]),
        seed=seed,
        include_bias=bool(fisher_config["include_bias"]),
        include_batchnorm_affine=bool(fisher_config["include_batchnorm_affine"]),
    )
    forget_fisher, forget_metadata = compute_diagonal_fisher(
        baseline.original_model,
        data.x_forget,
        data.y_forget,
        device=device,
        sample_size=min(
            int(fisher_config["fisher_forget_sample_size"]), len(data.x_forget)
        ),
        batch_size=int(fisher_config["fisher_batch_size"]),
        seed=seed + 1,
        include_bias=bool(fisher_config["include_bias"]),
        include_batchnorm_affine=bool(fisher_config["include_batchnorm_affine"]),
    )
    synchronize_device(device)
    core_time = time.perf_counter() - shared_start
    return _FisherPhase(
        teacher_logits=teacher_logits,
        retain_fisher=retain_fisher,
        forget_fisher=forget_fisher,
        retain_metadata=retain_metadata,
        forget_metadata=forget_metadata,
        core_time=float(core_time),
        method_time=float(positive_class_weight_time + core_time),
    )


def _finalize_single_seed_search(
    seeded_config: Mapping[str, Any],
    *,
    seed: int,
    output_dir: Path,
    proposed_config_path: Path,
    device: torch.device,
    mode: str,
    run_started: float,
    baseline: _BaselinePhase,
    retraining: _RetrainingPhase,
    fisher: _FisherPhase,
    best_hybrid: dict[str, Any] | None,
    comparison: pd.DataFrame,
    all_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Seleziona il finalista e persiste tutte le evidenze compatte del seed."""
    comparison.to_csv(output_dir / "search_comparison.csv", index=False)

    print(f"[seed {seed} | 6/7] Selezione deterministica dei finalisti")
    finalist_results = [retraining.search_result]
    if best_hybrid is not None:
        best_hybrid["method"] = "hybrid_fisher_dampening"
        finalist_results.append(best_hybrid)
    selected_result = select_best_search_result(
        finalist_results,
        baseline_precision_at_10=baseline.precision_at_10,
        utility_floor_ratio=float(seeded_config["utility_floor_ratio"]),
    )
    selected_method = str(selected_result["method"])
    if selected_method == "retraining_from_scratch":
        final_config = _build_retraining_final_config(
            seeded_config, seed=seed, best_epoch=retraining.best_epoch
        )
    else:
        final_config = _build_hybrid_final_config(
            seeded_config, selected_result, seed=seed
        )
    # Validate again immediately before writing the user-reviewable proposal.
    final_config = validate_final_config(final_config)
    write_json(proposed_config_path, final_config)

    finalist_rows = [
        _finalist_row(
            result,
            seed=seed,
            selected=result is selected_result,
        )
        for result in finalist_results
    ]
    finalists = pd.DataFrame(finalist_rows).sort_values(
        [
            "selected",
            "valid",
            "utility_floor_pass",
            "local_search_score",
            "local_privacy_proxy",
            "precision_at_10",
            "execution_time_seconds",
            "configuration",
        ],
        ascending=[False, False, False, False, False, False, True, True],
        kind="mergesort",
    )
    finalists.to_csv(output_dir / "finalists.csv", index=False)
    _write_best_histories(output_dir, best_hybrid)
    best_summary = {
        "seed": seed,
        "device": str(device),
        "selected_method": selected_method,
        "selected_configuration": selected_result["config"]["name"],
        "original_model_metrics": baseline.metrics,
        "utility_floor_precision_at_10": retraining.utility_floor,
        "selection_priority": [
            "valid",
            "utility_floor_pass",
            "local_search_score",
            "local_privacy_proxy",
            "precision_at_10",
            "execution_time_seconds",
            "configuration_name",
            "config_index",
        ],
        "selected": _result_summary(selected_result),
        "best_hybrid": _result_summary(best_hybrid),
        "retraining_finalist": {
            "status": "success",
            "valid": True,
            "effective_configuration": retraining.search_result["config"],
            "selected_epoch": retraining.best_epoch,
            "metrics": retraining.metrics,
        },
        "fisher": {
            "retain": fisher.retain_metadata,
            "forget": fisher.forget_metadata,
        },
    }
    write_json(output_dir / "best_candidate_summary.json", best_summary)

    success_count = sum(result.get("valid", False) for result in all_results)
    run_metadata = {
        "status": "completed",
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "effective_search_config": seeded_config,
        "original_model_metrics": baseline.metrics,
        "utility_floor_precision_at_10": retraining.utility_floor,
        "privacy_proxy_name": "local_retrained_reference_privacy_proxy",
        "privacy_proxy_limitation": (
            "Non equivale alla Membership Inference Attack ufficiale nascosta."
        ),
        "privacy_proxy_cv_auc": float(retraining.privacy_proxy.cv_auc),
        "privacy_proxy_reverse_direction": bool(retraining.privacy_proxy.reverse),
        "privacy_proxy_original_membership_mean": float(
            retraining.privacy_proxy.original_membership_mean
        ),
        "privacy_proxy_retrained_membership_mean": float(
            retraining.privacy_proxy.retrained_membership_mean
        ),
        "privacy_proxy_original_reference_distance": float(
            retraining.privacy_proxy.original_reference_distance
        ),
        "positive_class_weight_time_seconds": retraining.positive_class_weight_time,
        "shared_teacher_and_fisher_time_seconds": fisher.core_time,
        "shared_hybrid_method_time_seconds": fisher.method_time,
        "retain_fisher": fisher.retain_metadata,
        "forget_fisher": fisher.forget_metadata,
        "retraining_selection_time_seconds": retraining.selection_time,
        "retraining_fixed_replay_time_seconds": retraining.fixed_time,
        "retraining_best_epoch": retraining.best_epoch,
        "selected_method": selected_method,
        "selected_configuration": selected_result["config"]["name"],
        "candidate_count": len(all_results),
        "successful_candidate_count": int(success_count),
        "failed_candidate_count": int(len(all_results) - success_count),
        "total_search_wall_time_seconds": time.perf_counter() - run_started,
    }
    write_json(output_dir / "search_metadata.json", run_metadata)
    print(f"[seed {seed} | 7/7] Ricerca completata: {selected_method}")
    return {
        "seed": seed,
        "final_config": final_config,
        "selected_method": selected_method,
        "selected_configuration": selected_result["config"]["name"],
        "finalists": finalists,
        "metadata": run_metadata,
    }


def _run_single_seed(
    search_config: Mapping[str, Any],
    *,
    seed: int,
    data_dir: Path,
    output_dir: Path,
    proposed_config_path: Path,
    device: torch.device,
    mode: str,
) -> dict[str, Any]:
    """Run one isolated seed and persist all compact audit evidence."""
    run_started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_owned_run_files(output_dir, proposed_config_path)
    seeded_config = deepcopy(dict(search_config))
    seeded_config["seed"] = validate_seed(seed)
    write_json(output_dir / "effective_search_config.json", seeded_config)
    evaluation_batch_size = int(seeded_config["evaluation_batch_size"])

    baseline = _load_baseline_phase(
        seed=seed,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        evaluation_batch_size=evaluation_batch_size,
        validation_fraction=float(seeded_config["validation_fraction"]),
    )
    data = baseline.data

    retraining = _run_retraining_phase(
        seeded_config,
        baseline=baseline,
        output_dir=output_dir,
        device=device,
        seed=seed,
        evaluation_batch_size=evaluation_batch_size,
    )

    fisher = _compute_fisher_phase(
        seeded_config,
        baseline=baseline,
        positive_class_weight_time=retraining.positive_class_weight_time,
        device=device,
        seed=seed,
    )

    def model_builder(state_dict: Mapping[str, torch.Tensor] | None) -> torch.nn.Module:
        return build_model(
            baseline.artifact["architecture"], state_dict=state_dict, device=device
        )

    print(
        f"[seed {seed} | 5/7] Ricerca di {len(seeded_config['candidates'])} "
        "configurazioni base"
    )
    execute_kwargs = {
        "model_builder": model_builder,
        "original_state": baseline.original_state,
        "retain_fisher": fisher.retain_fisher,
        "forget_fisher": fisher.forget_fisher,
        "retain_features": data.x_retain_train,
        "retain_targets": data.y_retain_train,
        "retain_teacher_logits": fisher.teacher_logits,
        "validation_features": data.x_validation,
        "validation_targets": data.y_validation,
        "forget_features": data.x_forget,
        "forget_targets": data.y_forget,
        "privacy_proxy": retraining.privacy_proxy,
        "baseline_precision_at_10": baseline.precision_at_10,
        "retraining_time_seconds": retraining.fixed_time,
        "positive_class_weights": retraining.positive_class_weights,
        "device": device,
        "seed": seed,
        "shared_method_time_seconds": fisher.method_time,
        "evaluation_batch_size": evaluation_batch_size,
    }
    best_hybrid, comparison, all_results = progressive_search(
        merge_candidate_configs(seeded_config),
        execute_kwargs=execute_kwargs,
        baseline_precision_at_10=baseline.precision_at_10,
        utility_floor_ratio=float(seeded_config["utility_floor_ratio"]),
        add_gradient_ascent_variants=int(
            seeded_config["add_gradient_ascent_variants"]
        ),
    )
    return _finalize_single_seed_search(
        seeded_config,
        seed=seed,
        output_dir=output_dir,
        proposed_config_path=proposed_config_path,
        device=device,
        mode=mode,
        run_started=run_started,
        baseline=baseline,
        retraining=retraining,
        fisher=fisher,
        best_hybrid=best_hybrid,
        comparison=comparison,
        all_results=all_results,
    )


def _write_failed_seed(
    output_dir: Path, *, seed: int, mode: str, error: Exception
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "mode": mode,
        "seed": seed,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    write_json(output_dir / "search_metadata.json", payload)
    return payload


def main(raw_arguments: list[str] | None = None) -> int:
    """Esegue una ricerca singola o una semplice valutazione multi-seed."""
    raw_arguments = list(sys.argv[1:] if raw_arguments is None else raw_arguments)
    arguments = build_parser().parse_args(raw_arguments)
    output_dir, proposed_config = _resolve_output_paths(arguments, raw_arguments)
    _assert_safe_proposed_path(
        proposed_config, allow_canonical=arguments.allow_canonical_overwrite
    )
    loaded_config = _load_search_config(arguments.config)
    effective_config = build_effective_search_config(
        loaded_config,
        quick=bool(arguments.quick),
        max_candidates=arguments.max_candidates,
    )
    seeds = [
        validate_seed(seed)
        for seed in (arguments.seeds or [int(effective_config["seed"])])
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds non accetta duplicati.")
    device = _resolve_device(arguments.device)
    mode = "quick" if arguments.quick else "full"
    _print_startup_summary(
        arguments=arguments,
        output_dir=output_dir,
        proposed_config=proposed_config,
        effective_config=effective_config,
        seeds=seeds,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_multi_seed_files(output_dir)

    if len(seeds) == 1:
        try:
            _run_single_seed(
                effective_config,
                seed=seeds[0],
                data_dir=arguments.data_dir,
                output_dir=output_dir,
                proposed_config_path=proposed_config,
                device=device,
                mode=mode,
            )
        except Exception as error:
            _write_failed_seed(output_dir, seed=seeds[0], mode=mode, error=error)
            raise
        print(f"Configurazione proposta scritta in: {proposed_config}")
        return 0

    _clear_owned_run_files(output_dir, proposed_config)
    write_json(
        output_dir / "effective_search_config.json",
        {"mode": mode, "seeds": seeds, "config": effective_config},
    )
    successful_runs: list[dict[str, Any]] = []
    seed_failures: list[dict[str, Any]] = []
    finalist_frames: list[pd.DataFrame] = []
    multi_start = time.perf_counter()
    for seed in seeds:
        seed_output = output_dir / f"seed_{seed}"
        try:
            run = _run_single_seed(
                effective_config,
                seed=seed,
                data_dir=arguments.data_dir,
                output_dir=seed_output,
                proposed_config_path=seed_output / "proposed_final_config.json",
                device=device,
                mode=mode,
            )
            successful_runs.append(run)
            finalist_frames.append(run["finalists"])
        except Exception as error:  # noqa: BLE001 - other seeds can still run
            failure = _write_failed_seed(
                seed_output, seed=seed, mode=mode, error=error
            )
            seed_failures.append(failure)
            print(f"[seed {seed}] FAILED {type(error).__name__}: {error}")

    if seed_failures:
        pd.DataFrame(seed_failures).to_csv(
            output_dir / "seed_failures.csv", index=False
        )
    if not successful_runs:
        write_json(
            output_dir / "search_metadata.json",
            {
                "status": "failed",
                "mode": mode,
                "device": str(device),
                "seeds": seeds,
                "successful_seed_count": 0,
                "failed_seed_count": len(seed_failures),
                "total_wall_time_seconds": time.perf_counter() - multi_start,
            },
        )
        raise RuntimeError("Nessun seed ha completato la ricerca.")

    multi_seed_results = pd.concat(finalist_frames, ignore_index=True)
    multi_seed_results.to_csv(output_dir / "multi_seed_results.csv", index=False)
    summary = summarize_multi_seed_results(multi_seed_results)
    summary.to_csv(output_dir / "multi_seed_summary.csv", index=False)

    # Multi-seed evidence is advisory: keep the first requested successful seed as
    # the reproducible proposal and require inspection of the aggregate table.
    primary_run = min(
        successful_runs,
        key=lambda run: seeds.index(int(run["seed"])),
    )
    aggregate_proposal = deepcopy(primary_run["final_config"])
    aggregate_proposal["selection_note"] = (
        f"Proposta del seed primario {primary_run['seed']}; verificare "
        "multi_seed_summary.csv prima della promozione. La proxy locale non "
        "equivale alla MIA ufficiale nascosta."
    )
    aggregate_proposal = validate_final_config(aggregate_proposal)
    write_json(proposed_config, aggregate_proposal)
    write_json(
        output_dir / "search_metadata.json",
        {
            "status": "completed",
            "mode": mode,
            "device": str(device),
            "seeds": seeds,
            "effective_search_config": effective_config,
            "successful_seeds": [run["seed"] for run in successful_runs],
            "failed_seeds": [failure["seed"] for failure in seed_failures],
            "successful_seed_count": len(successful_runs),
            "failed_seed_count": len(seed_failures),
            "proposal_source_seed": primary_run["seed"],
            "proposal_policy": (
                "Seed primario riproducibile; la stabilita' e' riportata separatamente."
            ),
            "total_wall_time_seconds": time.perf_counter() - multi_start,
        },
    )
    print(f"Riepilogo multi-seed scritto in: {output_dir / 'multi_seed_summary.csv'}")
    print(f"Configurazione proposta scritta in: {proposed_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
