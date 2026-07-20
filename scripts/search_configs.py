"""Esegue baseline, ricerca ibrida e proposta della configurazione finale."""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.data import (  # noqa: E402
    load_challenge_data,
    save_validation_ids,
    validate_data_model_compatibility,
)
from machine_unlearning.metrics import (  # noqa: E402
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
from machine_unlearning.training import (  # noqa: E402
    compute_positive_class_weights,
    seed_everything,
    train_with_early_stopping,
)
from machine_unlearning.unlearning import (  # noqa: E402
    compute_diagonal_fisher,
    precompute_teacher_logits,
    progressive_search,
)
from machine_unlearning.workflow import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Definisce gli argomenti della ricerca, separati dal workflow finale."""
    parser = argparse.ArgumentParser(
        description="Cerca configurazioni di unlearning e propone final_config.json."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/search_configs.json")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/search"))
    parser.add_argument(
        "--selected-config",
        type=Path,
        default=Path("configs/final_config.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Limita i candidati per prove rapide; omettere per la ricerca completa.",
    )
    return parser


def _load_search_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "seed",
        "validation_fraction",
        "retraining",
        "fisher",
        "common_candidate",
        "candidates",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Configurazione di ricerca incompleta: {sorted(missing)}")
    return config


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA richiesto ma non disponibile.")
    return device


def _merge_candidate_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rende espliciti i valori comuni in ogni candidato eseguito."""
    common = dict(config["common_candidate"])
    return [{**common, **candidate} for candidate in config["candidates"]]


def main() -> int:
    """Esegue la ricerca e scrive risultati e configurazione proposta."""
    arguments = build_parser().parse_args()
    search_config = _load_search_config(arguments.config)
    seed = int(search_config["seed"])
    device = _resolve_device(arguments.device)
    output_dir = arguments.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    print("[1/7] Caricamento dati e modello originale")
    data = load_challenge_data(
        arguments.data_dir,
        validation_fraction=float(search_config["validation_fraction"]),
        seed=seed,
    )
    save_validation_ids(data, output_dir / "validation_ids.csv")
    artifact = load_model_artifact(arguments.data_dir / "model_artifact")
    validate_data_model_compatibility(data, artifact["architecture"])
    original_model = build_model_from_artifact(artifact, device=device)
    original_state = model_state_to_cpu(original_model)
    evaluation_batch_size = int(search_config["evaluation_batch_size"])
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
    baseline_precision = float(original_validation["precision_at_10"])

    print("[2/7] Retraining di riferimento con early stopping")
    retraining_config = search_config["retraining"]
    seed_everything(seed)
    retrained_model = build_model(artifact["architecture"], device=device)
    positive_class_weights = compute_positive_class_weights(
        data.y_retain_train, device=device
    )
    retraining_result = train_with_early_stopping(
        retrained_model,
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
    retraining_result.history.to_csv(output_dir / "retraining_history.csv", index=False)
    retrained_forget = evaluate_model(
        retraining_result.model,
        data.x_forget,
        data.y_forget,
        device=device,
        batch_size=evaluation_batch_size,
    )

    print("[3/7] Costruzione della proxy privacy out-of-fold")
    privacy_proxy = fit_reference_privacy_proxy(
        original_forget["logits"],
        retrained_forget["logits"],
        data.y_forget,
        seed=seed,
    )

    fisher_config = search_config["fisher"]
    print("[4/7] Teacher logits e Fisher condivise dalla ricerca")
    shared_start = time.perf_counter()
    teacher_logits = precompute_teacher_logits(
        original_model,
        data.x_retain_train,
        device=device,
        batch_size=int(fisher_config["teacher_batch_size"]),
    )
    retain_fisher, retain_fisher_metadata = compute_diagonal_fisher(
        original_model,
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
    forget_fisher, forget_fisher_metadata = compute_diagonal_fisher(
        original_model,
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
    shared_method_time = time.perf_counter() - shared_start

    def model_builder(state_dict: Mapping[str, torch.Tensor] | None) -> torch.nn.Module:
        return build_model(
            artifact["architecture"], state_dict=state_dict, device=device
        )

    candidates = _merge_candidate_configs(search_config)
    if arguments.max_candidates is not None:
        if arguments.max_candidates <= 0:
            raise ValueError("--max-candidates deve essere positivo.")
        candidates = candidates[: arguments.max_candidates]
    print(f"[5/7] Ricerca di {len(candidates)} configurazioni base")
    execute_kwargs = {
        "model_builder": model_builder,
        "original_state": original_state,
        "retain_fisher": retain_fisher,
        "forget_fisher": forget_fisher,
        "retain_features": data.x_retain_train,
        "retain_targets": data.y_retain_train,
        "retain_teacher_logits": teacher_logits,
        "validation_features": data.x_validation,
        "validation_targets": data.y_validation,
        "forget_features": data.x_forget,
        "forget_targets": data.y_forget,
        "privacy_proxy": privacy_proxy,
        "baseline_precision_at_10": baseline_precision,
        "retraining_time_seconds": retraining_result.elapsed_seconds,
        "positive_class_weights": positive_class_weights,
        "device": device,
        "seed": seed,
        "shared_method_time_seconds": shared_method_time,
    }
    best_hybrid, comparison, all_results = progressive_search(
        candidates,
        execute_kwargs=execute_kwargs,
        baseline_precision_at_10=baseline_precision,
        utility_floor_ratio=float(search_config["utility_floor_ratio"]),
        add_gradient_ascent_variants=int(search_config["add_gradient_ascent_variants"]),
    )
    comparison.to_csv(output_dir / "search_comparison.csv", index=False)

    print("[6/7] Confronto con il retraining da zero")
    retraining_metrics = evaluate_unlearning_candidate(
        retraining_result.model,
        validation_features=data.x_validation,
        validation_targets=data.y_validation,
        forget_features=data.x_forget,
        forget_targets=data.y_forget,
        device=device,
        privacy_proxy=privacy_proxy,
        baseline_precision_at_10=baseline_precision,
        retraining_time_seconds=retraining_result.elapsed_seconds,
        execution_time_seconds=retraining_result.elapsed_seconds,
        batch_size=evaluation_batch_size,
    )
    utility_floor = baseline_precision * float(search_config["utility_floor_ratio"])
    retraining_metrics.update(
        {
            "execution_time_seconds": retraining_result.elapsed_seconds,
            "best_epoch": retraining_result.best_epoch,
            "utility_floor_pass": retraining_metrics["precision_at_10"]
            >= utility_floor,
        }
    )
    hybrid_metrics = best_hybrid["metrics"]
    if (
        retraining_metrics["utility_floor_pass"]
        and not hybrid_metrics["utility_floor_pass"]
    ):
        selected_method = "retraining_from_scratch"
    elif (
        hybrid_metrics["utility_floor_pass"]
        and not retraining_metrics["utility_floor_pass"]
    ):
        selected_method = "hybrid_fisher_dampening"
    elif (
        retraining_metrics["local_search_score"] > hybrid_metrics["local_search_score"]
    ):
        selected_method = "retraining_from_scratch"
    else:
        selected_method = "hybrid_fisher_dampening"

    finalists = pd.DataFrame(
        [
            {
                "method": "hybrid_fisher_dampening",
                "configuration": best_hybrid["config"]["name"],
                **hybrid_metrics,
            },
            {
                "method": "retraining_from_scratch",
                "configuration": "retraining_reference",
                **retraining_metrics,
            },
        ]
    )
    finalists.to_csv(output_dir / "finalists.csv", index=False)

    if selected_method == "retraining_from_scratch":
        final_config = {
            "schema_version": 1,
            "name": f"retraining_from_scratch_fixed_{retraining_result.best_epoch}_epochs",
            "method": selected_method,
            "seed": seed,
            "validation_fraction": float(search_config["validation_fraction"]),
            "optimizer": retraining_config["optimizer"],
            "learning_rate": retraining_config["learning_rate"],
            "weight_decay": retraining_config["weight_decay"],
            "training_batch_size": retraining_config["training_batch_size"],
            "fixed_epochs": int(retraining_result.best_epoch),
            "gradient_clip": None,
            "evaluation_batch_size": evaluation_batch_size,
            "selection_note": "Selezionato dalla ricerca locale; la privacy e' una proxy, non la MIA ufficiale.",
        }
    else:
        final_config = {
            "schema_version": 1,
            "method": selected_method,
            "seed": seed,
            "validation_fraction": float(search_config["validation_fraction"]),
            "evaluation_batch_size": evaluation_batch_size,
            **deepcopy(fisher_config),
            **deepcopy(best_hybrid["config"]),
            "fixed_repair_epochs": int(best_hybrid["metrics"]["best_epoch"]),
            "selection_note": "Selezionato dalla ricerca locale; la privacy e' una proxy, non la MIA ufficiale.",
        }
    write_json(arguments.selected_config, final_config)
    write_json(
        output_dir / "search_metadata.json",
        {
            "privacy_proxy_cv_auc": privacy_proxy.cv_auc,
            "shared_method_time_seconds": shared_method_time,
            "retain_fisher": retain_fisher_metadata,
            "forget_fisher": forget_fisher_metadata,
            "selected_method": selected_method,
            "candidate_count": len(all_results),
        },
    )
    print("[7/7] Ricerca completata")
    print(f"Metodo proposto: {selected_method}")
    print(f"Configurazione scritta in: {arguments.selected_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
