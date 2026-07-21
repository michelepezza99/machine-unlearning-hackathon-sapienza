"""Deterministic multi-seed selection of a reviewable hybrid configuration."""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .search import merge_candidate_configs, validate_search_config
from .search_aggregation import configuration_fingerprint
from .training import validate_seed
from .workflow import validate_final_config


LOCAL_PROXY_LIMITATION = (
    "La proxy locale rispetto al retraining non equivale alla Membership "
    "Inference Attack ufficiale nascosta; la valutazione ufficiale resta pendente."
)

HYBRID_CANDIDATE_RUNTIME_FIELDS = (
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

SUMMARY_REQUIRED_COLUMNS = {
    "configuration_name",
    "configuration_fingerprint",
    "seed_count",
    "valid_rate",
    "utility_floor_pass_rate",
    "local_privacy_proxy_min",
    "local_privacy_proxy_mean",
    "precision_at_10_min",
    "utility_ratio_min",
    "execution_time_seconds_mean",
    "execution_time_seconds_max",
    "selected_parameter_fraction_mean",
    "best_epoch_mode",
}

RAW_REQUIRED_COLUMNS = {
    "seed",
    "configuration_name",
    "configuration_fingerprint",
    "configuration_fingerprint_scope",
    "evidence_mode",
    "valid",
    "utility_floor_pass",
    "best_epoch",
    "local_privacy_proxy",
    "precision_at_10",
    "utility_ratio",
    "execution_time_seconds",
    "selected_parameter_fraction",
    *HYBRID_CANDIDATE_RUNTIME_FIELDS,
}

RAW_RECOMPUTED_SUMMARY_FIELDS = (
    "seed_count",
    "valid_rate",
    "utility_floor_pass_rate",
    "local_privacy_proxy_min",
    "local_privacy_proxy_mean",
    "precision_at_10_min",
    "utility_ratio_min",
    "execution_time_seconds_mean",
    "execution_time_seconds_max",
    "selected_parameter_fraction_mean",
    "best_epoch_mode",
)

RANKING_AGGREGATE_FIELDS = (
    "local_privacy_proxy_min",
    "local_privacy_proxy_mean",
    "precision_at_10_min",
    "utility_ratio_min",
    "execution_time_seconds_mean",
    "execution_time_seconds_max",
    "selected_parameter_fraction_mean",
)

RANKING_SPECIFICATION = (
    ("local_privacy_proxy_min", "descending"),
    ("local_privacy_proxy_mean", "descending"),
    ("precision_at_10_min", "descending"),
    ("utility_ratio_min", "descending"),
    ("execution_time_seconds_mean", "ascending"),
    ("execution_time_seconds_max", "ascending"),
    ("selected_parameter_fraction_mean", "ascending"),
    ("top_fraction", "ascending_complexity"),
    ("gradient_ascent_steps", "ascending_complexity"),
    ("fixed_repair_epochs", "ascending_complexity"),
    ("recalibrate_batchnorm", "false_first_complexity"),
    ("selected_parameter_fraction_mean", "ascending_complexity"),
    ("configuration_name", "ascending_tiebreak"),
    ("configuration_fingerprint", "ascending_tiebreak"),
)


@dataclass(frozen=True)
class RecommendationResult:
    """Pure recommendation output before any files are written."""

    recommendation: dict[str, Any]
    final_config: dict[str, Any] | None
    diagnostics: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame, required: set[str], *, source: str
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Colonne mancanti in {source}: {missing}")


def _boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "si", "sì"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"{field} contiene un booleano non riconosciuto: {value!r}")


def _finite_float(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} deve essere numerico, ricevuto {value!r}.") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} deve essere finito, ricevuto {value!r}.")
    return number


def select_positive_epoch_mode(values: Iterable[Any]) -> tuple[int, list[int], dict[int, int]]:
    """Select a positive statistical mode after counting every observed epoch.

    Zero participates in the mode calculation but can never be selected.  If
    zero is the unique mode, or every modal value is non-positive, no safe
    fixed repair epoch can be inferred.  Tied positive modes are resolved
    toward the smaller value.
    """
    observed: list[int] = []
    for raw_value in values:
        number = _finite_float(raw_value, field="best_epoch")
        integer = int(number)
        if not math.isclose(number, integer, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"best_epoch deve essere intero, ricevuto {raw_value!r}.")
        if integer < 0:
            raise ValueError(
                f"best_epoch non puo' essere negativo, ricevuto {raw_value!r}."
            )
        observed.append(integer)
    if not observed:
        raise ValueError("Nessun best_epoch osservato per il candidato selezionato.")
    counts = Counter(observed)
    highest_count = max(counts.values())
    modal_values = [
        epoch for epoch, count in counts.items() if count == highest_count
    ]
    positive_modes = [epoch for epoch in modal_values if epoch > 0]
    if not positive_modes:
        raise ValueError(
            "Nessuna moda positiva di best_epoch: non e' sicuro inventare "
            "fixed_repair_epochs."
        )
    selected = min(positive_modes)
    return selected, observed, dict(sorted(counts.items()))


def _search_config_candidates(
    search_config: Mapping[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Return accepted ``(name, full fingerprint)`` candidate identities."""

    shared = {
        "validation_fraction": search_config["validation_fraction"],
        "evaluation_batch_size": search_config["evaluation_batch_size"],
        "utility_floor_ratio": search_config["utility_floor_ratio"],
        **dict(search_config["fisher"]),
    }
    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    for candidate in merge_candidate_configs(search_config):
        name = str(candidate["name"]).strip()
        effective = {**shared, **candidate}
        fingerprint = configuration_fingerprint(effective)
        accepted[(name, fingerprint)] = candidate
    return accepted


def _numeric_values(
    values: Iterable[Any],
    *,
    field: str,
    allow_missing: bool,
) -> np.ndarray:
    parsed: list[float] = []
    for value in values:
        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
            if allow_missing:
                continue
            raise ValueError(f"{field} non puo' essere mancante nelle righe valide.")
        number = _finite_float(value, field=field)
        parsed.append(number)
    return np.asarray(parsed, dtype=np.float64)


def _recompute_raw_summary(group: pd.DataFrame) -> dict[str, float | int]:
    """Recompute mandatory rates and ranking aggregates from raw evidence."""

    if group.empty:
        raise ValueError("Nessuna riga raw associata al fingerprint del summary.")
    valid_values = [_boolean(value, field="valid") for value in group["valid"]]
    floor_values = [
        _boolean(value, field="utility_floor_pass")
        for value in group["utility_floor_pass"]
    ]
    valid_mask = np.asarray(valid_values, dtype=bool)
    valid_group = group.loc[valid_mask]
    run_count = len(group)
    valid_count = int(valid_mask.sum())
    floor_count = sum(
        valid and floor
        for valid, floor in zip(valid_values, floor_values, strict=True)
    )

    normalized_seeds: set[int] = set()
    for raw_seed in group["seed"]:
        number = _finite_float(raw_seed, field="seed")
        seed = int(number)
        if not math.isclose(number, seed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"seed deve essere intero, ricevuto {raw_seed!r}.")
        normalized_seeds.add(validate_seed(seed))

    metric_values = {
        metric: _numeric_values(
            valid_group[metric],
            field=metric,
            allow_missing=metric == "selected_parameter_fraction",
        )
        for metric in (
            "local_privacy_proxy",
            "precision_at_10",
            "utility_ratio",
            "execution_time_seconds",
            "selected_parameter_fraction",
        )
    }

    def statistic(metric: str, operation: str) -> float:
        values = metric_values[metric]
        if not len(values):
            return float("nan")
        if operation == "mean":
            return float(values.mean())
        if operation == "min":
            return float(values.min())
        if operation == "max":
            return float(values.max())
        raise AssertionError(f"Statistica sconosciuta: {operation}")

    valid_epochs: list[int] = []
    for raw_epoch in valid_group["best_epoch"]:
        number = _finite_float(raw_epoch, field="best_epoch")
        epoch = int(number)
        if not math.isclose(number, epoch, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"best_epoch deve essere intero, ricevuto {raw_epoch!r}."
            )
        if epoch < 0:
            raise ValueError(
                f"best_epoch non puo' essere negativo, ricevuto {raw_epoch!r}."
            )
        valid_epochs.append(epoch)
    if valid_epochs:
        counts = Counter(valid_epochs)
        maximum_count = max(counts.values())
        raw_epoch_mode: float | int = min(
            epoch for epoch, count in counts.items() if count == maximum_count
        )
    else:
        raw_epoch_mode = float("nan")

    return {
        "seed_count": len(normalized_seeds),
        "valid_rate": float(valid_count / run_count),
        "utility_floor_pass_rate": float(floor_count / run_count),
        "local_privacy_proxy_min": statistic("local_privacy_proxy", "min"),
        "local_privacy_proxy_mean": statistic("local_privacy_proxy", "mean"),
        "precision_at_10_min": statistic("precision_at_10", "min"),
        "utility_ratio_min": statistic("utility_ratio", "min"),
        "execution_time_seconds_mean": statistic(
            "execution_time_seconds", "mean"
        ),
        "execution_time_seconds_max": statistic("execution_time_seconds", "max"),
        "selected_parameter_fraction_mean": statistic(
            "selected_parameter_fraction", "mean"
        ),
        "best_epoch_mode": raw_epoch_mode,
    }


def _aggregate_values_match(summary_value: Any, raw_value: Any) -> bool:
    try:
        summary_number = float(summary_value)
        raw_number = float(raw_value)
    except (TypeError, ValueError):
        return False
    if math.isnan(summary_number) and math.isnan(raw_number):
        return True
    if not math.isfinite(summary_number) or not math.isfinite(raw_number):
        return False
    return math.isclose(
        summary_number,
        raw_number,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def _crosscheck_summary_with_raw(
    summary_row: pd.Series,
    raw_aggregates: Mapping[str, float | int],
) -> tuple[bool, bool, list[str]]:
    mismatches = [
        field
        for field in RAW_RECOMPUTED_SUMMARY_FIELDS
        if not _aggregate_values_match(summary_row[field], raw_aggregates[field])
    ]
    ranking_matches = not any(
        field in RANKING_AGGREGATE_FIELDS for field in mismatches
    )
    return not mismatches, ranking_matches, mismatches


def _representative_raw_rows(
    raw: pd.DataFrame,
) -> dict[str, pd.Series]:
    representatives: dict[str, pd.Series] = {}
    for fingerprint, group in raw.groupby("configuration_fingerprint", sort=True):
        ordered = group.sort_values(
            ["configuration_name", "seed"], kind="mergesort"
        )
        representatives[str(fingerprint)] = ordered.iloc[0]
    return representatives


def _coerce_summary_numeric(summary: pd.DataFrame) -> pd.DataFrame:
    converted = summary.copy()
    numeric_columns = sorted(
        SUMMARY_REQUIRED_COLUMNS
        - {"configuration_name", "configuration_fingerprint"}
    )
    for column in numeric_columns:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")
    return converted


def _raw_constraint_status(
    group: pd.DataFrame, expected_seeds: tuple[int, ...]
) -> tuple[bool, bool, bool, str]:
    try:
        normalized_seeds: list[int] = []
        for raw_seed in group["seed"].tolist():
            number = _finite_float(raw_seed, field="seed")
            seed = int(number)
            if not math.isclose(number, seed, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"seed deve essere intero, ricevuto {raw_seed!r}.")
            normalized_seeds.append(validate_seed(seed))
        seeds = tuple(sorted(normalized_seeds))
    except (TypeError, ValueError) as error:
        return False, False, False, f"seed non numerico: {error}"
    seed_set_matches = seeds == expected_seeds
    try:
        all_valid = all(_boolean(value, field="valid") for value in group["valid"])
        all_floor = all(
            _boolean(value, field="utility_floor_pass")
            for value in group["utility_floor_pass"]
        )
    except ValueError as error:
        return seed_set_matches, False, False, str(error)
    return seed_set_matches, all_valid, all_floor, ""


def _candidate_rank(
    summary_row: pd.Series,
    raw_row: pd.Series,
    *,
    epoch_mode: int | None,
) -> tuple[Any, ...]:
    def descending(column: str) -> float:
        value = _finite_float(summary_row[column], field=column)
        return -value

    def ascending(column: str, *, missing: float = math.inf) -> float:
        try:
            value = float(summary_row[column])
        except (TypeError, ValueError):
            return missing
        return value if math.isfinite(value) else missing

    runtime_config = _candidate_runtime_config(raw_row)
    top_fraction = float(runtime_config["top_fraction"])
    gradient_steps = int(runtime_config["gradient_ascent_steps"])
    recalibrate = bool(runtime_config["recalibrate_batchnorm"])
    return (
        descending("local_privacy_proxy_min"),
        descending("local_privacy_proxy_mean"),
        descending("precision_at_10_min"),
        descending("utility_ratio_min"),
        ascending("execution_time_seconds_mean"),
        ascending("execution_time_seconds_max"),
        ascending("selected_parameter_fraction_mean"),
        top_fraction,
        gradient_steps,
        epoch_mode if epoch_mode is not None else math.inf,
        1 if recalibrate else 0,
        ascending("selected_parameter_fraction_mean"),
        str(summary_row["configuration_name"]),
        str(summary_row["configuration_fingerprint"]),
    )


def _python_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _candidate_runtime_config(row: pd.Series) -> dict[str, Any]:
    integers = {
        "gradient_ascent_steps",
        "gradient_ascent_batch_size",
        "repair_batch_size",
        "batchnorm_recalibration_batch_size",
    }
    booleans = {"freeze_selected_during_repair", "recalibrate_batchnorm"}
    config: dict[str, Any] = {}
    for field in HYBRID_CANDIDATE_RUNTIME_FIELDS:
        value = row[field]
        if field in booleans:
            config[field] = _boolean(value, field=field)
        elif field in integers:
            number = _finite_float(value, field=field)
            integer = int(number)
            if not math.isclose(number, integer, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"{field} deve essere intero, ricevuto {value!r}.")
            config[field] = integer
        else:
            config[field] = _finite_float(value, field=field)
    return config


def build_hybrid_final_config(
    search_config: Mapping[str, Any],
    candidate_row: pd.Series,
    *,
    seed: int,
    fixed_repair_epochs: int,
) -> dict[str, Any]:
    """Build and validate the non-canonical hybrid configuration."""
    validate_search_config(search_config)
    selected_seed = validate_seed(seed)
    if fixed_repair_epochs < 1:
        raise ValueError("fixed_repair_epochs deve essere positivo.")
    fisher = deepcopy(dict(search_config["fisher"]))
    candidate = _candidate_runtime_config(candidate_row)
    name = str(candidate_row["configuration_name"]).strip()
    if not name:
        raise ValueError("configuration_name deve essere una stringa non vuota.")
    config: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "method": "hybrid_fisher_dampening",
        "seed": selected_seed,
        "validation_fraction": float(search_config["validation_fraction"]),
        "evaluation_batch_size": int(search_config["evaluation_batch_size"]),
        **fisher,
        **candidate,
        "fixed_repair_epochs": int(fixed_repair_epochs),
        "selection_note": (
            "Migliore configurazione ibrida fra i candidati valutati localmente "
            "sui seed richiesti secondo filtri rigidi e ranking gerarchico. "
            + LOCAL_PROXY_LIMITATION
        ),
        "selection_status": "provisional_local_proxy_official_mia_pending",
    }
    return validate_final_config(config)


def recommend_hybrid_configuration(
    summary: pd.DataFrame,
    raw: pd.DataFrame,
    search_config: Mapping[str, Any],
    *,
    expected_seeds: Sequence[int],
    final_seed: int,
) -> RecommendationResult:
    """Apply strict completeness filters and deterministic hierarchical ranking."""
    _require_columns(summary, SUMMARY_REQUIRED_COLUMNS, source="summary")
    _require_columns(raw, RAW_REQUIRED_COLUMNS, source="raw all-candidate results")
    validate_search_config(search_config)
    normalized_expected = tuple(sorted(validate_seed(seed) for seed in expected_seeds))
    if not normalized_expected:
        raise ValueError("expected_seeds non puo' essere vuoto.")
    if len(normalized_expected) != len(set(normalized_expected)):
        raise ValueError("expected_seeds non accetta duplicati.")
    selected_seed = validate_seed(final_seed)
    if selected_seed not in normalized_expected:
        raise ValueError("final_seed deve appartenere al set dei seed attesi.")

    converted_summary = _coerce_summary_numeric(summary)
    accepted_candidates = _search_config_candidates(search_config)
    representatives = _representative_raw_rows(raw)
    diagnostic_entries: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    ranked_candidates: list[
        tuple[
            tuple[Any, ...],
            pd.Series,
            pd.Series,
            int | None,
            dict[str, float | int],
        ]
    ] = []

    for _, row in converted_summary.sort_values(
        ["configuration_name", "configuration_fingerprint"], kind="mergesort"
    ).iterrows():
        fingerprint = str(row["configuration_fingerprint"])
        candidate_raw = raw.loc[
            raw["configuration_fingerprint"].astype(str) == fingerprint
        ].copy()
        representative = representatives.get(fingerprint)
        raw_seed_pass, raw_valid_pass, raw_floor_pass, raw_error = (
            _raw_constraint_status(candidate_raw, normalized_expected)
        )
        summary_name = str(row["configuration_name"])
        raw_names = {
            str(value) for value in candidate_raw["configuration_name"].tolist()
        }
        raw_name_pass = raw_names == {summary_name}
        identity_error = (
            "configuration_name del summary non coincide con le righe raw."
            if not raw_name_pass
            else ""
        )
        accepted_candidate = accepted_candidates.get((summary_name, fingerprint))
        search_config_membership_pass = accepted_candidate is not None
        membership_error = (
            "La coppia configuration_name/fingerprint non appartiene ai candidati "
            "semantici della search config fornita."
            if not search_config_membership_pass
            else ""
        )
        scopes = {
            str(value).strip().casefold()
            for value in candidate_raw["configuration_fingerprint_scope"].tolist()
        }
        full_scope_pass = scopes == {"candidate_and_shared"}
        scope_error = (
            "configuration_fingerprint_scope deve essere candidate_and_shared "
            f"per ogni seed, ricevuto {sorted(scopes)}."
            if not full_scope_pass
            else ""
        )
        evidence_modes = {
            str(value).strip().casefold()
            for value in candidate_raw["evidence_mode"].tolist()
        }
        full_evidence_mode_pass = evidence_modes == {"full"}
        evidence_mode_error = (
            "evidence_mode deve essere full per ogni seed, ricevuto "
            f"{sorted(evidence_modes)}."
            if not full_evidence_mode_pass
            else ""
        )

        raw_aggregates: dict[str, float | int] | None = None
        aggregate_error = ""
        summary_raw_match = False
        ranking_aggregates_match = False
        aggregate_mismatches: list[str] = []
        try:
            raw_aggregates = _recompute_raw_summary(candidate_raw)
            (
                summary_raw_match,
                ranking_aggregates_match,
                aggregate_mismatches,
            ) = _crosscheck_summary_with_raw(row, raw_aggregates)
            if aggregate_mismatches:
                aggregate_error = (
                    "Summary non coerente con gli aggregati raw ricalcolati: "
                    f"{aggregate_mismatches}."
                )
        except (KeyError, TypeError, ValueError) as error:
            aggregate_error = f"aggregati raw non calcolabili: {error}"

        summary_seed_pass = bool(
            raw_aggregates is not None
            and _aggregate_values_match(
                row["seed_count"], raw_aggregates["seed_count"]
            )
            and int(raw_aggregates["seed_count"]) == len(normalized_expected)
        )
        summary_valid_pass = bool(
            raw_aggregates is not None
            and _aggregate_values_match(
                row["valid_rate"], raw_aggregates["valid_rate"]
            )
            and math.isclose(
                float(raw_aggregates["valid_rate"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        summary_floor_pass = bool(
            raw_aggregates is not None
            and _aggregate_values_match(
                row["utility_floor_pass_rate"],
                raw_aggregates["utility_floor_pass_rate"],
            )
            and math.isclose(
                float(raw_aggregates["utility_floor_pass_rate"]),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        epoch_mode: int | None = None
        epoch_error = ""
        try:
            epoch_mode, _, _ = select_positive_epoch_mode(candidate_raw["best_epoch"])
        except ValueError as error:
            epoch_error = str(error)
        candidate_rank: tuple[Any, ...] | None = None
        ranking_error = ""
        candidate_for_config: pd.Series | None = None
        if accepted_candidate is not None:
            candidate_for_config = pd.Series(
                {"configuration_name": summary_name, **accepted_candidate}
            )
        if (
            representative is not None
            and candidate_for_config is not None
            and raw_aggregates is not None
        ):
            try:
                ranking_row = row.copy()
                for field in RANKING_AGGREGATE_FIELDS:
                    ranking_row[field] = raw_aggregates[field]
                candidate_rank = _candidate_rank(
                    ranking_row, candidate_for_config, epoch_mode=epoch_mode
                )
            except (KeyError, TypeError, ValueError) as error:
                ranking_error = f"ranking non calcolabile: {error}"
        eligible = bool(
            representative is not None
            and raw_name_pass
            and search_config_membership_pass
            and full_scope_pass
            and full_evidence_mode_pass
            and summary_raw_match
            and ranking_aggregates_match
            and summary_seed_pass
            and summary_valid_pass
            and summary_floor_pass
            and raw_seed_pass
            and raw_valid_pass
            and raw_floor_pass
            and epoch_mode is not None
            and candidate_rank is not None
        )
        diagnostics = {
            "configuration_name": summary_name,
            "configuration_fingerprint": fingerprint,
            "configuration_identity_pass": raw_name_pass,
            "search_config_membership_pass": search_config_membership_pass,
            "full_fingerprint_scope_pass": full_scope_pass,
            "full_evidence_mode_pass": full_evidence_mode_pass,
            "summary_raw_evidence_match": summary_raw_match,
            "ranking_aggregates_match": ranking_aggregates_match,
            "expected_seed_count_pass": summary_seed_pass,
            "expected_seed_set_pass": raw_seed_pass,
            "valid_rate_pass": summary_valid_pass and raw_valid_pass,
            "utility_floor_pass_rate_pass": summary_floor_pass and raw_floor_pass,
            "positive_epoch_available": epoch_mode is not None,
            "ranking_values_valid": candidate_rank is not None,
            "eligible": eligible,
            "constraint_error": "; ".join(
                message
                for message in (
                    identity_error,
                    membership_error,
                    scope_error,
                    evidence_mode_error,
                    raw_error,
                    aggregate_error,
                    epoch_error,
                    ranking_error,
                )
                if message
            ),
        }
        for column, _ in RANKING_SPECIFICATION[:7]:
            if column in row:
                diagnostics[column] = _python_scalar(row[column])
                if raw_aggregates is not None and column in raw_aggregates:
                    diagnostics[f"raw_recomputed_{column}"] = _python_scalar(
                        raw_aggregates[column]
                    )
        constraint_passes = (
            raw_name_pass,
            search_config_membership_pass,
            full_scope_pass,
            full_evidence_mode_pass,
            summary_raw_match,
            ranking_aggregates_match,
            summary_seed_pass,
            raw_seed_pass,
            summary_valid_pass and raw_valid_pass,
            summary_floor_pass and raw_floor_pass,
            epoch_mode is not None,
            candidate_rank is not None,
        )
        fallback_rank = candidate_rank or (
            *(math.inf for _ in range(12)),
            summary_name,
            fingerprint,
        )
        diagnostic_entries.append(
            (
                (
                    0 if eligible else 1,
                    sum(not passed for passed in constraint_passes),
                    *fallback_rank,
                ),
                diagnostics,
            )
        )
        if (
            eligible
            and candidate_for_config is not None
            and candidate_rank is not None
            and raw_aggregates is not None
        ):
            ranked_candidates.append(
                (
                    candidate_rank,
                    row,
                    candidate_for_config,
                    epoch_mode,
                    raw_aggregates,
                )
            )

    diagnostic_entries.sort(key=lambda item: item[0])
    diagnostic_rows = [
        {"diagnostic_rank": rank, **diagnostics}
        for rank, (_, diagnostics) in enumerate(diagnostic_entries, start=1)
    ]
    diagnostics_frame = pd.DataFrame(diagnostic_rows)

    base_recommendation: dict[str, Any] = {
        "schema_version": 1,
        "privacy_evaluator": "local_retrained_reference_privacy_proxy",
        "privacy_proxy_limitation": LOCAL_PROXY_LIMITATION,
        "official_mia_status": "pending",
        "expected_seeds": list(normalized_expected),
        "final_seed": selected_seed,
        "mandatory_filters": {
            "seed_count": len(normalized_expected),
            "exact_seed_set": list(normalized_expected),
            "valid_rate": 1.0,
            "utility_floor_pass_rate": 1.0,
            "configuration_fingerprint_scope": "candidate_and_shared",
            "evidence_mode": "full",
            "search_config_candidate_membership": True,
            "summary_matches_raw_recomputed_aggregates": True,
        },
        "ranking": [
            {"field": field, "direction": direction}
            for field, direction in RANKING_SPECIFICATION
        ],
        "candidate_count": int(len(converted_summary)),
        "eligible_candidate_count": int(len(ranked_candidates)),
    }

    if not ranked_candidates:
        failed_constraints: dict[str, int] = {}
        if not diagnostics_frame.empty:
            for column in (
                "configuration_identity_pass",
                "search_config_membership_pass",
                "full_fingerprint_scope_pass",
                "full_evidence_mode_pass",
                "summary_raw_evidence_match",
                "ranking_aggregates_match",
                "expected_seed_count_pass",
                "expected_seed_set_pass",
                "valid_rate_pass",
                "utility_floor_pass_rate_pass",
                "positive_epoch_available",
                "ranking_values_valid",
            ):
                failed_constraints[column] = int(
                    (~diagnostics_frame[column].astype(bool)).sum()
                )
        recommendation = {
            **base_recommendation,
            "status": "provisional_no_safe_selection",
            "final_config_generated": False,
            "failed_constraint_counts": failed_constraints,
            "message": (
                "Nessun candidato soddisfa tutti i vincoli obbligatori; i vincoli "
                "non sono stati rilassati. Consultare la diagnostica ordinata."
            ),
        }
        return RecommendationResult(recommendation, None, diagnostics_frame)

    ranked_candidates.sort(key=lambda item: item[0])
    (
        _,
        selected_summary,
        selected_candidate,
        fixed_epochs,
        selected_raw_aggregates,
    ) = ranked_candidates[0]
    if fixed_epochs is None:
        raise AssertionError("Un candidato eleggibile deve avere un'epoca positiva.")
    fingerprint = str(selected_summary["configuration_fingerprint"])
    selected_name = str(selected_summary["configuration_name"])
    selected_rows = raw.loc[
        (raw["configuration_fingerprint"].astype(str) == fingerprint)
        & (raw["configuration_name"].astype(str) == selected_name)
    ].sort_values("seed", kind="mergesort")
    fixed_epochs, observed_epochs, epoch_counts = select_positive_epoch_mode(
        selected_rows["best_epoch"]
    )
    final_config = build_hybrid_final_config(
        search_config,
        selected_candidate,
        seed=final_seed,
        fixed_repair_epochs=fixed_epochs,
    )
    recommendation = {
        **base_recommendation,
        "status": "selected_provisional_official_mia_pending",
        "final_config_generated": True,
        "selected": {
            "configuration_name": str(selected_summary["configuration_name"]),
            "configuration_fingerprint": fingerprint,
            "summary_metrics": {
                column: _python_scalar(selected_summary[column])
                for column in sorted(converted_summary.columns)
                if column.endswith(("_mean", "_std", "_min", "_max", "_rate"))
                and pd.notna(selected_summary[column])
            },
            "raw_recomputed_ranking_metrics": {
                column: _python_scalar(selected_raw_aggregates[column])
                for column in RANKING_AGGREGATE_FIELDS
            },
            "fixed_repair_epochs": {
                "observed_by_seed": [
                    {
                        "seed": int(row["seed"]),
                        "best_epoch": int(float(row["best_epoch"])),
                    }
                    for _, row in selected_rows.iterrows()
                ],
                "observed_values": observed_epochs,
                "value_counts": {
                    str(epoch): count for epoch, count in epoch_counts.items()
                },
                "selected": fixed_epochs,
                "rule": (
                    "La moda e' calcolata includendo zero; fra le mode positive "
                    "a pari frequenza sceglie la minore. Una moda unica pari a "
                    "zero non e' selezionabile."
                ),
            },
        },
        "selection_note": final_config["selection_note"],
    }
    return RecommendationResult(recommendation, final_config, diagnostics_frame)


def assert_noncanonical_config_path(path: str | Path, *, repository_root: Path) -> None:
    """Refuse the canonical final configuration regardless of caller intent."""
    destination = Path(path).resolve()
    canonical = (repository_root / "configs" / "final_config.json").resolve()
    if destination == canonical:
        raise ValueError(
            "La raccomandazione ibrida non puo' scrivere configs/final_config.json."
        )
