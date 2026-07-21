"""Small, deterministic helpers for the experimental search workflow."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .training import validate_seed


MULTI_SEED_METRICS = (
    "precision_at_10",
    "validation_bce",
    "forget_bce",
    "local_privacy_proxy",
    "execution_time_seconds",
    "local_search_score",
)

QUICK_RETRAINING_EPOCHS = 2
QUICK_FISHER_SAMPLE_SIZE = 32
QUICK_CANDIDATE_COUNT = 2


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"{key} deve essere un oggetto JSON.")
    return value


def _finite_number(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> float:
    if key not in config or isinstance(config[key], bool):
        raise ValueError(f"{key} deve essere un numero.")
    try:
        value = float(config[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} deve essere un numero.") from error
    if not math.isfinite(value):
        raise ValueError(f"{key} deve essere finito.")
    if minimum is not None:
        valid = value >= minimum if minimum_inclusive else value > minimum
        if not valid:
            relation = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{key} deve essere {relation} {minimum}.")
    if maximum is not None:
        valid = value <= maximum if maximum_inclusive else value < maximum
        if not valid:
            relation = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{key} deve essere {relation} {maximum}.")
    return value


def _integer(
    config: Mapping[str, Any],
    key: str,
    *,
    minimum: int,
) -> int:
    if key not in config or isinstance(config[key], bool):
        raise ValueError(f"{key} deve essere un intero >= {minimum}.")
    value = config[key]
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"{key} deve essere un intero >= {minimum}.")
    return value


def _boolean(config: Mapping[str, Any], key: str) -> bool:
    if key not in config or not isinstance(config[key], bool):
        raise ValueError(f"{key} deve essere true o false.")
    return bool(config[key])


def merge_candidate_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand shared candidate values without mutating the loaded JSON."""
    common = dict(_mapping(config, "common_candidate"))
    if "utility_floor_ratio" in config:
        common["utility_floor_ratio"] = config["utility_floor_ratio"]
    candidates = config.get("candidates")
    if not isinstance(candidates, list):
        raise TypeError("candidates deve essere una lista JSON.")
    merged: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"candidates[{index}] deve essere un oggetto JSON.")
        merged.append({**common, **dict(candidate)})
    return merged


def validate_search_config(config: Mapping[str, Any]) -> None:
    """Fail early with actionable errors before expensive search operations."""
    required = {
        "schema_version",
        "seed",
        "validation_fraction",
        "evaluation_batch_size",
        "utility_floor_ratio",
        "add_gradient_ascent_variants",
        "retraining",
        "fisher",
        "common_candidate",
        "candidates",
    }
    missing = required - set(config)
    if missing:
        raise KeyError(f"Configurazione di ricerca incompleta: {sorted(missing)}")

    if _integer(config, "schema_version", minimum=1) != 1:
        raise ValueError("schema_version della ricerca deve essere 1.")
    validate_seed(_integer(config, "seed", minimum=0))
    _finite_number(
        config,
        "validation_fraction",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
        maximum_inclusive=False,
    )
    _integer(config, "evaluation_batch_size", minimum=1)
    utility_floor_ratio = _finite_number(
        config, "utility_floor_ratio", minimum=0.0, maximum=1.0
    )
    _integer(config, "add_gradient_ascent_variants", minimum=0)

    retraining = _mapping(config, "retraining")
    optimizer = retraining.get("optimizer")
    if not isinstance(optimizer, str) or optimizer.lower() not in {"adam", "adamw", "sgd"}:
        raise ValueError("retraining.optimizer deve essere adam, adamw o sgd.")
    _finite_number(
        retraining, "learning_rate", minimum=0.0, minimum_inclusive=False
    )
    _finite_number(retraining, "weight_decay", minimum=0.0)
    _integer(retraining, "training_batch_size", minimum=1)
    _integer(retraining, "max_epochs", minimum=1)
    _integer(retraining, "patience", minimum=1)
    if "momentum" in retraining:
        _finite_number(retraining, "momentum", minimum=0.0)

    fisher = _mapping(config, "fisher")
    for key in (
        "teacher_batch_size",
        "fisher_retain_sample_size",
        "fisher_forget_sample_size",
        "fisher_batch_size",
    ):
        _integer(fisher, key, minimum=1)
    _boolean(fisher, "include_bias")
    _boolean(fisher, "include_batchnorm_affine")
    if "quick" in config:
        quick = _mapping(config, "quick")
        for key in (
            "retraining_max_epochs",
            "retraining_patience",
            "fisher_retain_sample_size",
            "fisher_forget_sample_size",
            "candidate_count",
            "repair_max_epochs",
            "repair_patience",
        ):
            _integer(quick, key, minimum=1)

    candidates = merge_candidate_configs(config)
    if not candidates:
        raise ValueError("Serve almeno un candidato di ricerca.")
    names: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        prefix = f"candidates[{index - 1}]"
        name = candidate.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{prefix}.name deve essere una stringa non vuota.")
        names.append(name)
        _finite_number(
            candidate,
            "top_fraction",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _finite_number(
            candidate,
            "forget_absolute_quantile",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _finite_number(
            candidate,
            "minimum_dampening_factor",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        )
        _finite_number(candidate, "dampening_strength", minimum=0.0, maximum=1.0)
        _finite_number(
            candidate, "fisher_ratio_power", minimum=0.0, minimum_inclusive=False
        )
        _finite_number(
            candidate, "repair_learning_rate", minimum=0.0, minimum_inclusive=False
        )
        _finite_number(candidate, "repair_weight_decay", minimum=0.0)
        _integer(candidate, "repair_batch_size", minimum=1)
        _integer(candidate, "repair_max_epochs", minimum=1)
        _integer(candidate, "repair_patience", minimum=1)
        for key in (
            "supervised_loss_weight",
            "distillation_weight",
            "parameter_regularization_weight",
            "selected_parameter_weight",
            "gradient_ascent_retain_distillation_weight",
        ):
            _finite_number(candidate, key, minimum=0.0)
        _finite_number(
            candidate, "gradient_clip", minimum=0.0, minimum_inclusive=False
        )
        _boolean(candidate, "freeze_selected_during_repair")
        _integer(candidate, "gradient_ascent_steps", minimum=0)
        _finite_number(
            candidate,
            "gradient_ascent_learning_rate",
            minimum=0.0,
            minimum_inclusive=False,
        )
        _integer(candidate, "gradient_ascent_batch_size", minimum=1)
        _boolean(candidate, "recalibrate_batchnorm")
        _integer(candidate, "batchnorm_recalibration_batch_size", minimum=1)
        candidate_floor = _finite_number(
            candidate, "utility_floor_ratio", minimum=0.0, maximum=1.0
        )
        if not math.isclose(candidate_floor, utility_floor_ratio, abs_tol=1e-12):
            raise ValueError(
                f"{prefix}.utility_floor_ratio deve coincidere col valore globale."
            )
    if len(names) != len(set(names)):
        raise ValueError("I nomi dei candidati devono essere univoci.")


def build_effective_search_config(
    config: Mapping[str, Any],
    *,
    quick: bool,
    max_candidates: int | None,
) -> dict[str, Any]:
    """Apply visible quick/limit overrides before any expensive computation."""
    effective = deepcopy(dict(config))
    if max_candidates is not None and max_candidates <= 0:
        raise ValueError("--max-candidates deve essere positivo.")
    if quick:
        quick_settings = dict(effective.get("quick", {}))
        retraining = dict(_mapping(effective, "retraining"))
        retraining["max_epochs"] = min(
            int(retraining["max_epochs"]),
            int(quick_settings.get("retraining_max_epochs", QUICK_RETRAINING_EPOCHS)),
        )
        retraining["patience"] = min(
            int(retraining["patience"]),
            int(quick_settings.get("retraining_patience", 1)),
        )
        effective["retraining"] = retraining

        fisher = dict(_mapping(effective, "fisher"))
        fisher["fisher_retain_sample_size"] = min(
            int(fisher["fisher_retain_sample_size"]),
            int(
                quick_settings.get(
                    "fisher_retain_sample_size", QUICK_FISHER_SAMPLE_SIZE
                )
            ),
        )
        fisher["fisher_forget_sample_size"] = min(
            int(fisher["fisher_forget_sample_size"]),
            int(
                quick_settings.get(
                    "fisher_forget_sample_size", QUICK_FISHER_SAMPLE_SIZE
                )
            ),
        )
        effective["fisher"] = fisher
        common = dict(_mapping(effective, "common_candidate"))
        common["repair_max_epochs"] = int(
            quick_settings.get("repair_max_epochs", 1)
        )
        common["repair_patience"] = int(quick_settings.get("repair_patience", 1))
        effective["common_candidate"] = common
        effective["add_gradient_ascent_variants"] = 0
        quick_candidates: list[Any] = []
        quick_candidate_count = int(
            quick_settings.get("candidate_count", QUICK_CANDIDATE_COUNT)
        )
        for candidate in list(effective["candidates"])[:quick_candidate_count]:
            if not isinstance(candidate, Mapping):
                # The normal validator will provide the indexed actionable error.
                quick_candidates.append(candidate)
                continue
            quick_candidate = deepcopy(dict(candidate))
            quick_candidate["repair_max_epochs"] = common["repair_max_epochs"]
            quick_candidate["repair_patience"] = common["repair_patience"]
            quick_candidates.append(quick_candidate)
        effective["candidates"] = quick_candidates

    if max_candidates is not None:
        effective["candidates"] = list(effective["candidates"])[:max_candidates]
    # Candidate early stopping and final ranking must use one authoritative floor.
    common = dict(_mapping(effective, "common_candidate"))
    common["utility_floor_ratio"] = effective["utility_floor_ratio"]
    effective["common_candidate"] = common
    validate_search_config(effective)
    return effective


def summarize_multi_seed_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successful finalist metrics with population statistics."""
    required = {"seed", "method", "configuration", "valid", *MULTI_SEED_METRICS}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Colonne multi-seed mancanti: {sorted(missing)}")
    successful = results.loc[results["valid"].astype(bool)].copy()
    for metric in MULTI_SEED_METRICS:
        successful = successful.loc[np.isfinite(successful[metric].astype(float))]
    rows: list[dict[str, Any]] = []
    for (method, configuration), group in successful.groupby(
        ["method", "configuration"], sort=True, dropna=False
    ):
        row: dict[str, Any] = {
            "method": method,
            "configuration": configuration,
            "seed_count": int(group["seed"].nunique()),
        }
        for metric in MULTI_SEED_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)
