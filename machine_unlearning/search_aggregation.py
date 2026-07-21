"""Deterministic aggregation helpers for progressive search evidence.

The search runner stores one ``search_comparison.csv`` per seed.  This module
combines those tables without dropping failed candidates, assigns a semantic
configuration identity, computes population statistics, and exposes a small
Pareto analysis based on the local privacy proxy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CONFIGURATION_NAME_COLUMN = "configuration_name"
CONFIGURATION_FINGERPRINT_COLUMN = "configuration_fingerprint"
CONFIGURATION_SCOPE_COLUMN = "configuration_fingerprint_scope"

CANDIDATE_FLOAT_FIELDS = (
    "top_fraction",
    "forget_absolute_quantile",
    "minimum_dampening_factor",
    "dampening_strength",
    "fisher_ratio_power",
    "gradient_ascent_learning_rate",
    "gradient_ascent_retain_distillation_weight",
    "repair_learning_rate",
    "repair_weight_decay",
    "supervised_loss_weight",
    "distillation_weight",
    "parameter_regularization_weight",
    "selected_parameter_weight",
    "gradient_clip",
    "utility_floor_ratio",
)

CANDIDATE_INTEGER_FIELDS = (
    "gradient_ascent_steps",
    "gradient_ascent_batch_size",
    "repair_batch_size",
    "repair_max_epochs",
    "repair_patience",
    "batchnorm_recalibration_batch_size",
)

CANDIDATE_BOOLEAN_FIELDS = (
    "freeze_selected_during_repair",
    "recalibrate_batchnorm",
)

SHARED_FLOAT_FIELDS = (
    "validation_fraction",
)

SHARED_INTEGER_FIELDS = (
    "evaluation_batch_size",
    "teacher_batch_size",
    "fisher_retain_sample_size",
    "fisher_forget_sample_size",
    "fisher_batch_size",
)

SHARED_BOOLEAN_FIELDS = (
    "include_bias",
    "include_batchnorm_affine",
)

# Keep this order stable: it is also the order used for configuration columns
# carried into the aggregate summary.
SEMANTIC_CONFIGURATION_FIELDS = (
    "validation_fraction",
    "evaluation_batch_size",
    "utility_floor_ratio",
    "teacher_batch_size",
    "fisher_retain_sample_size",
    "fisher_forget_sample_size",
    "fisher_batch_size",
    "include_bias",
    "include_batchnorm_affine",
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
    "repair_max_epochs",
    "repair_patience",
    "supervised_loss_weight",
    "distillation_weight",
    "parameter_regularization_weight",
    "selected_parameter_weight",
    "gradient_clip",
    "freeze_selected_during_repair",
    "recalibrate_batchnorm",
    "batchnorm_recalibration_batch_size",
)

SUMMARY_METRICS = (
    "precision_at_10",
    "utility_ratio",
    "validation_bce",
    "forget_bce",
    "local_privacy_proxy",
    "execution_time_seconds",
    "local_search_score",
    "selected_parameter_fraction",
)

# ``utility_ratio`` can be absent when every candidate in a search run failed:
# the current failure-row schema contains only the six ranking metrics.
FAILURE_SAFE_REQUIRED_METRICS = (
    "precision_at_10",
    "validation_bce",
    "forget_bce",
    "local_privacy_proxy",
    "execution_time_seconds",
    "local_search_score",
    "selected_parameter_fraction",
    "best_epoch",
)

REQUIRED_INPUT_COLUMNS = frozenset(
    {
        "seed",
        "valid",
        "utility_floor_pass",
        *CANDIDATE_FLOAT_FIELDS,
        *CANDIDATE_INTEGER_FIELDS,
        *CANDIDATE_BOOLEAN_FIELDS,
        *FAILURE_SAFE_REQUIRED_METRICS,
    }
)

OPTIONAL_BOOLEAN_COLUMNS = (
    "gradient_ascent_used",
    "batchnorm_recalibration_used",
    "selected",
)

LOCAL_PRIVACY_DISCLAIMER = (
    "La proxy locale retrained-reference non equivale alla Membership Inference "
    "Attack ufficiale nascosta e non costituisce validazione privacy ufficiale."
)

_SEED_DIRECTORY_PATTERN = re.compile(r"seed_(\d+)$")


@dataclass(frozen=True)
class SearchResultFile:
    """One per-seed comparison table and its optional effective config."""

    seed: int
    comparison_path: Path
    effective_config_path: Path | None
    search_metadata_path: Path | None


@dataclass(frozen=True)
class AggregationResult:
    """All deterministic tables and metadata produced by one aggregation."""

    raw: pd.DataFrame
    summary: pd.DataFrame
    pareto: pd.DataFrame
    metadata: dict[str, Any]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def normalize_boolean(value: Any, *, column: str = "value") -> bool:
    """Parse only genuine booleans or case-insensitive ``true``/``false``.

    In particular, this avoids the unsafe ``astype(bool)`` behavior where the
    non-empty string ``"False"`` becomes true.
    """

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        f"{column} deve contenere soltanto valori booleani true/false; "
        f"ricevuto {value!r}."
    )


def _parse_float(
    value: Any,
    *,
    column: str,
    allow_missing: bool = False,
) -> float | None:
    if _is_missing(value) or (isinstance(value, str) and not value.strip()):
        if allow_missing:
            return None
        raise ValueError(f"{column} non puo' essere mancante.")
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{column} deve essere numerico, non booleano.")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} contiene un valore non numerico: {value!r}.") from error
    if not math.isfinite(number):
        if allow_missing and math.isnan(number):
            return None
        raise ValueError(f"{column} deve contenere valori finiti; ricevuto {value!r}.")
    # Make equivalent textual encodings such as -0 and 0 fingerprint identically.
    return 0.0 if number == 0.0 else number


def _parse_integer(
    value: Any,
    *,
    column: str,
    allow_missing: bool = False,
) -> int | None:
    number = _parse_float(value, column=column, allow_missing=allow_missing)
    if number is None:
        return None
    if not float(number).is_integer():
        raise ValueError(f"{column} deve contenere interi; ricevuto {value!r}.")
    return int(number)


def _typed_semantic_value(field: str, value: Any) -> bool | int | float:
    if field in CANDIDATE_BOOLEAN_FIELDS or field in SHARED_BOOLEAN_FIELDS:
        return normalize_boolean(value, column=field)
    if field in CANDIDATE_INTEGER_FIELDS or field in SHARED_INTEGER_FIELDS:
        parsed_integer = _parse_integer(value, column=field)
        assert parsed_integer is not None
        return parsed_integer
    if (
        field in CANDIDATE_FLOAT_FIELDS
        or field in SHARED_FLOAT_FIELDS
        or field == "utility_floor_ratio"
    ):
        parsed_float = _parse_float(value, column=field)
        assert parsed_float is not None
        return parsed_float
    raise KeyError(f"Campo semantico sconosciuto: {field}")


def canonical_effective_configuration(
    configuration: Mapping[str, Any],
) -> dict[str, bool | int | float]:
    """Return a typed semantic configuration with inactive knobs normalized.

    Names, rationales, paths, seeds, status fields, and metrics are excluded by
    construction.  GA learning-rate/batch knobs do not affect execution when
    ``gradient_ascent_steps`` is zero; the BatchNorm batch size is similarly
    inactive when recalibration is disabled.  Those inactive values are omitted.
    """

    canonical: dict[str, bool | int | float] = {}
    for field in SEMANTIC_CONFIGURATION_FIELDS:
        if field not in configuration or _is_missing(configuration[field]):
            continue
        canonical[field] = _typed_semantic_value(field, configuration[field])

    if int(canonical.get("gradient_ascent_steps", 0)) == 0:
        for field in (
            "gradient_ascent_learning_rate",
            "gradient_ascent_batch_size",
            "gradient_ascent_retain_distillation_weight",
        ):
            canonical.pop(field, None)
    if canonical.get("recalibrate_batchnorm") is False:
        canonical.pop("batchnorm_recalibration_batch_size", None)
    return canonical


def canonical_configuration_json(configuration: Mapping[str, Any]) -> str:
    """Serialize semantic configuration fields in one stable JSON form."""

    canonical = canonical_effective_configuration(configuration)
    return json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def configuration_fingerprint(configuration: Mapping[str, Any]) -> str:
    """Compute a stable SHA-256 identity for an effective configuration."""

    payload = canonical_configuration_json(configuration).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def discover_search_result_files(input_dir: str | Path) -> list[SearchResultFile]:
    """Discover ``seed_*/search_comparison.csv`` in deterministic seed order."""

    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Directory di input non trovata: {root}")
    paths = list(root.glob("seed_*/search_comparison.csv"))
    if not paths:
        raise FileNotFoundError(
            f"Nessun file seed_*/search_comparison.csv trovato in {root}."
        )

    discovered: list[SearchResultFile] = []
    for comparison_path in paths:
        match = _SEED_DIRECTORY_PATTERN.fullmatch(comparison_path.parent.name)
        if match is None:
            raise ValueError(
                "Directory seed non valida per il file "
                f"{comparison_path}: atteso seed_<intero non negativo>."
            )
        seed = int(match.group(1))
        effective_path = comparison_path.parent / "effective_search_config.json"
        metadata_path = comparison_path.parent / "search_metadata.json"
        discovered.append(
            SearchResultFile(
                seed=seed,
                comparison_path=comparison_path,
                effective_config_path=effective_path if effective_path.is_file() else None,
                search_metadata_path=metadata_path if metadata_path.is_file() else None,
            )
        )

    discovered.sort(key=lambda item: (item.seed, item.comparison_path.as_posix()))
    duplicate_directories: dict[int, list[str]] = {}
    for item in discovered:
        duplicate_directories.setdefault(item.seed, []).append(
            item.comparison_path.parent.name
        )
    duplicates = {
        seed: names for seed, names in duplicate_directories.items() if len(names) > 1
    }
    if duplicates:
        raise ValueError(f"Directory duplicate per lo stesso seed: {duplicates}")
    return discovered


def _read_json_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON non valido in {path} ({label}): {error}") from error
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} ({label}) deve contenere un oggetto JSON.")
    return payload


def validate_full_search_evidence(sources: Sequence[SearchResultFile]) -> str:
    """Require every discovered per-seed run to be completed in full mode."""

    if not sources:
        raise ValueError("Nessuna sorgente di ricerca da validare.")
    for source in sources:
        metadata_path = source.search_metadata_path
        if metadata_path is None:
            raise FileNotFoundError(
                f"Manca search_metadata.json per il seed {source.seed}; "
                "non e' possibile dimostrare che l'evidenza sia full e completa."
            )
        metadata = _read_json_mapping(metadata_path, label="metadati del seed")
        status = metadata.get("status")
        mode = metadata.get("mode")
        if status != "completed":
            raise ValueError(
                f"{metadata_path}: status deve essere 'completed', ricevuto {status!r}."
            )
        if mode != "full":
            raise ValueError(
                f"{metadata_path}: mode deve essere 'full'; evidenza {mode!r} "
                "non e' valida per aggregazione e raccomandazione."
            )
        if "seed" in metadata:
            metadata_seed = _parse_integer(
                metadata["seed"], column=f"seed in {metadata_path}"
            )
            if metadata_seed != source.seed:
                raise ValueError(
                    f"{metadata_path}: seed {metadata_seed} diverso dalla directory "
                    f"seed_{source.seed}."
                )
    return "full"


def infer_expected_seeds_from_root_config(input_dir: str | Path) -> list[int]:
    """Read the exact seed set from the root multi-seed effective config."""

    root = Path(input_dir)
    config_path = root / "effective_search_config.json"
    if not config_path.is_file():
        raise ValueError(
            f"Impossibile inferire i seed attesi: manca {config_path}. "
            "Specificare --expected-seeds esplicitamente."
        )
    payload = _read_json_mapping(config_path, label="configurazione multi-seed")
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError(
            f"{config_path} non contiene una lista non vuota 'seeds'; "
            "specificare --expected-seeds esplicitamente."
        )
    seeds: list[int] = []
    for value in raw_seeds:
        seed = _parse_integer(value, column=f"seeds in {config_path}")
        assert seed is not None
        if seed < 0:
            raise ValueError(f"{config_path}: i seed non possono essere negativi.")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"{config_path}: la lista seeds contiene duplicati.")
    return sorted(seeds)


def _load_shared_effective_settings(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json_mapping(path, label="configurazione effettiva del seed")
    # Also accept the wrapper written at the root of a multi-seed output.
    if isinstance(payload.get("config"), Mapping):
        payload = payload["config"]

    settings: dict[str, Any] = {}
    for field in ("validation_fraction", "evaluation_batch_size", "utility_floor_ratio"):
        if field in payload:
            settings[field] = _typed_semantic_value(field, payload[field])

    fisher = payload.get("fisher")
    if fisher is not None and not isinstance(fisher, Mapping):
        raise TypeError(f"{path}: fisher deve essere un oggetto JSON.")
    if isinstance(fisher, Mapping):
        for field in (*SHARED_INTEGER_FIELDS[1:], *SHARED_BOOLEAN_FIELDS):
            if field in fisher:
                settings[field] = _typed_semantic_value(field, fisher[field])
    return settings


def _require_input_columns(frame: pd.DataFrame, *, path: Path) -> None:
    missing = REQUIRED_INPUT_COLUMNS - set(frame.columns)
    if missing:
        raise KeyError(
            f"{path}: colonne obbligatorie mancanti: {sorted(missing)}"
        )
    if "name" not in frame.columns and CONFIGURATION_NAME_COLUMN not in frame.columns:
        raise KeyError(
            f"{path}: manca la colonna name (o {CONFIGURATION_NAME_COLUMN})."
        )


def _merge_semantic_values(
    candidate: Mapping[str, Any],
    shared: Mapping[str, Any],
    *,
    path: Path,
    row_number: int,
) -> dict[str, Any]:
    merged = dict(candidate)
    for field, value in shared.items():
        if field in merged and not _is_missing(merged[field]):
            current = _typed_semantic_value(field, merged[field])
            if current != value:
                raise ValueError(
                    f"{path}, riga {row_number}: {field} nel CSV ({current!r}) "
                    f"non coincide con effective_search_config.json ({value!r})."
                )
        merged[field] = value
    return merged


def _normalize_comparison_frame(
    source: SearchResultFile,
    *,
    input_root: Path,
) -> pd.DataFrame:
    try:
        frame = pd.read_csv(source.comparison_path, dtype=object)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"File CSV vuoto: {source.comparison_path}") from error
    _require_input_columns(frame, path=source.comparison_path)
    if frame.empty:
        raise ValueError(f"Nessun risultato candidato in {source.comparison_path}.")

    frame = frame.copy()
    if CONFIGURATION_NAME_COLUMN not in frame:
        frame[CONFIGURATION_NAME_COLUMN] = frame["name"]
    elif "name" in frame:
        left = frame[CONFIGURATION_NAME_COLUMN].astype(str).str.strip()
        right = frame["name"].astype(str).str.strip()
        if not left.equals(right):
            raise ValueError(
                f"{source.comparison_path}: name e {CONFIGURATION_NAME_COLUMN} "
                "non coincidono."
            )
    frame[CONFIGURATION_NAME_COLUMN] = frame[CONFIGURATION_NAME_COLUMN].map(
        lambda value: str(value).strip() if not _is_missing(value) else ""
    )
    if (frame[CONFIGURATION_NAME_COLUMN] == "").any():
        rows = (frame.index[frame[CONFIGURATION_NAME_COLUMN] == ""] + 2).tolist()
        raise ValueError(
            f"{source.comparison_path}: nomi configurazione vuoti alle righe {rows}."
        )

    normalized_seeds: list[int] = []
    for row_index, value in enumerate(frame["seed"], start=2):
        seed = _parse_integer(value, column=f"seed (riga {row_index})")
        assert seed is not None
        if seed != source.seed:
            raise ValueError(
                f"{source.comparison_path}, riga {row_index}: seed CSV {seed} "
                f"diverso dal seed della directory {source.seed}."
            )
        normalized_seeds.append(seed)
    frame["seed"] = normalized_seeds

    for column in ("valid", "utility_floor_pass", *CANDIDATE_BOOLEAN_FIELDS):
        frame[column] = [
            normalize_boolean(value, column=f"{column} (riga {row_index})")
            for row_index, value in enumerate(frame[column], start=2)
        ]
    for column in OPTIONAL_BOOLEAN_COLUMNS:
        if column in frame:
            frame[column] = [
                normalize_boolean(value, column=f"{column} (riga {row_index})")
                for row_index, value in enumerate(frame[column], start=2)
            ]

    for column in CANDIDATE_FLOAT_FIELDS:
        frame[column] = [
            _parse_float(value, column=f"{column} (riga {row_index})")
            for row_index, value in enumerate(frame[column], start=2)
        ]
    for column in CANDIDATE_INTEGER_FIELDS:
        frame[column] = [
            _parse_integer(value, column=f"{column} (riga {row_index})")
            for row_index, value in enumerate(frame[column], start=2)
        ]

    if "utility_ratio" not in frame:
        frame["utility_ratio"] = np.nan
    for column in SUMMARY_METRICS:
        allow_selected_missing = column == "selected_parameter_fraction"
        normalized: list[float] = []
        for row_index, (value, valid) in enumerate(
            zip(frame[column], frame["valid"], strict=True), start=2
        ):
            parsed = _parse_float(
                value,
                column=f"{column} (riga {row_index})",
                allow_missing=(not bool(valid)) or allow_selected_missing,
            )
            normalized.append(np.nan if parsed is None else parsed)
        frame[column] = normalized

    normalized_epochs: list[float | int] = []
    for row_index, (value, valid) in enumerate(
        zip(frame["best_epoch"], frame["valid"], strict=True), start=2
    ):
        parsed_epoch = _parse_integer(
            value,
            column=f"best_epoch (riga {row_index})",
            allow_missing=not bool(valid),
        )
        normalized_epochs.append(np.nan if parsed_epoch is None else parsed_epoch)
    frame["best_epoch"] = normalized_epochs
    if "config_index" in frame:
        frame["config_index"] = [
            _parse_integer(
                value,
                column=f"config_index (riga {row_index})",
                allow_missing=True,
            )
            for row_index, value in enumerate(frame["config_index"], start=2)
        ]

    shared = _load_shared_effective_settings(source.effective_config_path)
    for field in (
        *SHARED_FLOAT_FIELDS,
        *SHARED_INTEGER_FIELDS,
        *SHARED_BOOLEAN_FIELDS,
    ):
        if field not in frame:
            frame[field] = shared.get(field, np.nan)
        elif field in shared:
            normalized_values: list[Any] = []
            for row_index, value in enumerate(frame[field], start=2):
                if _is_missing(value):
                    normalized_values.append(shared[field])
                    continue
                current = _typed_semantic_value(field, value)
                if current != shared[field]:
                    raise ValueError(
                        f"{source.comparison_path}, riga {row_index}: {field} "
                        "non coincide con effective_search_config.json."
                    )
                normalized_values.append(shared[field])
            frame[field] = normalized_values
        elif field in frame:
            values: list[Any] = []
            for row_index, value in enumerate(frame[field], start=2):
                if _is_missing(value):
                    values.append(np.nan)
                else:
                    values.append(_typed_semantic_value(field, value))
            frame[field] = values

    canonical_json_values: list[str] = []
    fingerprints: list[str] = []
    scopes: list[str] = []
    shared_identity_fields = set(shared) - {"utility_floor_ratio"}
    for row_position, (_, row) in enumerate(frame.iterrows(), start=2):
        candidate = {
            field: row[field]
            for field in SEMANTIC_CONFIGURATION_FIELDS
            if field in row and not _is_missing(row[field])
        }
        effective = _merge_semantic_values(
            candidate,
            shared,
            path=source.comparison_path,
            row_number=row_position,
        )
        canonical_json = canonical_configuration_json(effective)
        canonical_json_values.append(canonical_json)
        fingerprints.append(hashlib.sha256(canonical_json.encode("utf-8")).hexdigest())
        row_has_shared = any(
            field in effective and not _is_missing(effective[field])
            for field in (
                "validation_fraction",
                "evaluation_batch_size",
                "teacher_batch_size",
                "fisher_retain_sample_size",
                "fisher_forget_sample_size",
                "fisher_batch_size",
                "include_bias",
                "include_batchnorm_affine",
            )
        )
        scope = (
            "candidate_and_shared"
            if shared_identity_fields or row_has_shared
            else "candidate_only"
        )
        scopes.append(scope)

    frame["effective_configuration_json"] = canonical_json_values
    frame[CONFIGURATION_FINGERPRINT_COLUMN] = fingerprints
    frame[CONFIGURATION_SCOPE_COLUMN] = scopes
    frame["run_outcome"] = [
        "invalid"
        if not valid
        else "valid_utility_pass"
        if utility_pass
        else "valid_utility_fail"
        for valid, utility_pass in zip(
            frame["valid"], frame["utility_floor_pass"], strict=True
        )
    ]
    try:
        relative_source = source.comparison_path.relative_to(input_root).as_posix()
    except ValueError:
        relative_source = source.comparison_path.as_posix()
    frame["source_file"] = relative_source
    frame["source_row"] = np.arange(2, len(frame) + 2, dtype=np.int64)
    return frame


def _stable_raw_column_order(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "seed",
        CONFIGURATION_NAME_COLUMN,
        CONFIGURATION_FINGERPRINT_COLUMN,
        CONFIGURATION_SCOPE_COLUMN,
        "effective_configuration_json",
        "name",
        "config_index",
        "status",
        "valid",
        "utility_floor_pass",
        "run_outcome",
        "evidence_mode",
        *SEMANTIC_CONFIGURATION_FIELDS,
        "error_type",
        "error_message",
        "selected_parameter_count",
        "selected_parameter_fraction",
        "gradient_ascent_used",
        "batchnorm_recalibration_used",
        *SUMMARY_METRICS,
        "best_epoch",
        "source_file",
        "source_row",
    ]
    ordered: list[str] = []
    for column in preferred:
        if column in frame and column not in ordered:
            ordered.append(column)
    ordered.extend(sorted(set(frame.columns) - set(ordered)))
    return ordered


def _validate_candidate_identities(frame: pd.DataFrame) -> None:
    duplicate_mask = frame.duplicated(
        subset=["seed", CONFIGURATION_FINGERPRINT_COLUMN], keep=False
    )
    if bool(duplicate_mask.any()):
        columns = [
            "seed",
            CONFIGURATION_NAME_COLUMN,
            CONFIGURATION_FINGERPRINT_COLUMN,
            "source_file",
            "source_row",
        ]
        duplicates = frame.loc[duplicate_mask, columns].to_dict(orient="records")
        raise ValueError(
            "Risultati candidati duplicati per seed e configurazione effettiva: "
            f"{duplicates}"
        )

    fingerprint_counts = frame.groupby(CONFIGURATION_NAME_COLUMN, sort=True)[
        CONFIGURATION_FINGERPRINT_COLUMN
    ].nunique()
    inconsistent_names = fingerprint_counts.loc[fingerprint_counts > 1].index.tolist()
    if inconsistent_names:
        details: dict[str, list[str]] = {}
        for name in inconsistent_names:
            details[str(name)] = sorted(
                frame.loc[
                    frame[CONFIGURATION_NAME_COLUMN] == name,
                    CONFIGURATION_FINGERPRINT_COLUMN,
                ].unique()
            )
        raise ValueError(
            "Lo stesso configuration_name identifica configurazioni effettive "
            f"incoerenti: {details}"
        )

    name_counts = frame.groupby(CONFIGURATION_FINGERPRINT_COLUMN, sort=True)[
        CONFIGURATION_NAME_COLUMN
    ].nunique()
    inconsistent_fingerprints = name_counts.loc[name_counts > 1].index.tolist()
    if inconsistent_fingerprints:
        aliases: dict[str, list[str]] = {}
        for fingerprint in inconsistent_fingerprints:
            aliases[str(fingerprint)] = sorted(
                frame.loc[
                    frame[CONFIGURATION_FINGERPRINT_COLUMN] == fingerprint,
                    CONFIGURATION_NAME_COLUMN,
                ]
                .astype(str)
                .unique()
            )
        raise ValueError(
            "La stessa configurazione_fingerprint e' associata a display name "
            f"incoerenti (alias): {aliases}"
        )


def load_all_candidate_results(
    input_dir: str | Path,
    *,
    require_full_evidence: bool = False,
) -> tuple[pd.DataFrame, list[SearchResultFile]]:
    """Load, normalize, fingerprint, validate, and stably order every row."""

    root = Path(input_dir)
    sources = discover_search_result_files(root)
    evidence_mode = validate_full_search_evidence(sources) if require_full_evidence else None
    frames = [
        _normalize_comparison_frame(source, input_root=root) for source in sources
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if evidence_mode is not None:
        combined["evidence_mode"] = evidence_mode
    _validate_candidate_identities(combined)

    sort_columns = ["seed", CONFIGURATION_NAME_COLUMN, CONFIGURATION_FINGERPRINT_COLUMN]
    if "config_index" in combined:
        combined["_config_index_sort"] = pd.to_numeric(
            combined["config_index"], errors="coerce"
        ).fillna(np.inf)
        sort_columns.insert(1, "_config_index_sort")
    combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    combined = combined.drop(columns=["_config_index_sort"], errors="ignore")
    return combined.loc[:, _stable_raw_column_order(combined)], sources


def _validated_expected_seeds(
    discovered_seeds: Sequence[int],
    *,
    expected_seeds: Sequence[int] | None,
    expected_seed_count: int | None,
) -> tuple[list[int] | None, int]:
    discovered = sorted(set(int(seed) for seed in discovered_seeds))
    if expected_seeds is not None:
        parsed = [
            _parse_integer(seed, column="expected_seeds") for seed in expected_seeds
        ]
        explicit = [int(seed) for seed in parsed if seed is not None]
        if len(explicit) != len(set(explicit)):
            raise ValueError("expected_seeds non accetta duplicati.")
        if any(seed < 0 for seed in explicit):
            raise ValueError("expected_seeds non accetta seed negativi.")
        explicit.sort()
        if expected_seed_count is not None and expected_seed_count != len(explicit):
            raise ValueError(
                "expected_seed_count non coincide con il numero di expected_seeds."
            )
        return explicit, len(explicit)

    if expected_seed_count is not None:
        if expected_seed_count < 1:
            raise ValueError("expected_seed_count deve essere positivo.")
        if len(discovered) > expected_seed_count:
            raise ValueError(
                f"Scoperti {len(discovered)} seed, oltre expected_seed_count="
                f"{expected_seed_count}."
            )
        return None, int(expected_seed_count)
    return discovered, len(discovered)


def _population_statistics(values: pd.Series) -> dict[str, float]:
    finite = values.dropna().to_numpy(dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {name: float("nan") for name in ("mean", "std", "min", "max")}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=0)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def summarize_candidate_results(
    raw_results: pd.DataFrame,
    *,
    expected_seeds: Sequence[int] | None = None,
    expected_seed_count: int | None = None,
) -> pd.DataFrame:
    """Aggregate candidates using valid-run metrics and population statistics."""

    required = {
        "seed",
        CONFIGURATION_NAME_COLUMN,
        CONFIGURATION_FINGERPRINT_COLUMN,
        CONFIGURATION_SCOPE_COLUMN,
        "effective_configuration_json",
        "valid",
        "utility_floor_pass",
        "best_epoch",
        *SUMMARY_METRICS,
    }
    missing = required - set(raw_results.columns)
    if missing:
        raise KeyError(f"Colonne mancanti nei risultati normalizzati: {sorted(missing)}")

    discovered = sorted(raw_results["seed"].astype(int).unique().tolist())
    expected_set, resolved_expected_count = _validated_expected_seeds(
        discovered,
        expected_seeds=expected_seeds,
        expected_seed_count=expected_seed_count,
    )
    rows: list[dict[str, Any]] = []
    grouped = raw_results.groupby(
        [CONFIGURATION_NAME_COLUMN, CONFIGURATION_FINGERPRINT_COLUMN],
        sort=True,
        dropna=False,
    )
    for (configuration_name, fingerprint), group in grouped:
        observed = sorted(group["seed"].astype(int).unique().tolist())
        missing_seed_values = (
            sorted(set(expected_set) - set(observed)) if expected_set is not None else []
        )
        missing_count = (
            len(missing_seed_values)
            if expected_set is not None
            else max(resolved_expected_count - len(observed), 0)
        )
        valid_mask = group["valid"].astype(bool)
        utility_pass_mask = valid_mask & group["utility_floor_pass"].astype(bool)
        valid_group = group.loc[valid_mask]
        run_count = int(len(group))
        valid_count = int(valid_mask.sum())
        pass_count = int(utility_pass_mask.sum())

        canonical_values = group["effective_configuration_json"].unique().tolist()
        if len(canonical_values) != 1:
            raise ValueError(
                f"Fingerprint {fingerprint} associato a JSON canonici diversi."
            )
        canonical = json.loads(canonical_values[0])
        scopes = sorted(group[CONFIGURATION_SCOPE_COLUMN].astype(str).unique())
        if len(scopes) != 1:
            raise ValueError(
                f"Fingerprint {fingerprint} associato a scope diversi: {scopes}."
            )

        row: dict[str, Any] = {
            CONFIGURATION_NAME_COLUMN: str(configuration_name),
            CONFIGURATION_FINGERPRINT_COLUMN: str(fingerprint),
            CONFIGURATION_SCOPE_COLUMN: scopes[0],
            **{field: canonical.get(field) for field in SEMANTIC_CONFIGURATION_FIELDS},
            "seed_count": len(observed),
            "run_count": run_count,
            "expected_seed_count": resolved_expected_count,
            "observed_seeds": json.dumps(observed, separators=(",", ":")),
            "missing_seed_count": missing_count,
            "missing_run_count": missing_count,
            "missing_seeds": json.dumps(missing_seed_values, separators=(",", ":")),
            "complete_seed_coverage": bool(missing_count == 0),
            "invalid_count": run_count - valid_count,
            "valid_count": valid_count,
            "valid_rate": float(valid_count / run_count),
            "utility_floor_pass_count": pass_count,
            "utility_floor_pass_rate": float(pass_count / run_count),
            "eligible_count": pass_count,
        }
        for metric in SUMMARY_METRICS:
            statistics = _population_statistics(valid_group[metric])
            for statistic, value in statistics.items():
                row[f"{metric}_{statistic}"] = value

        epoch_values = valid_group["best_epoch"].dropna().to_numpy(dtype=np.float64)
        epoch_values = epoch_values[np.isfinite(epoch_values)]
        if len(epoch_values):
            unique_epochs, counts = np.unique(epoch_values.astype(np.int64), return_counts=True)
            maximum_count = int(counts.max())
            mode = int(unique_epochs[counts == maximum_count].min())
            row.update(
                {
                    "best_epoch_mean": float(epoch_values.mean()),
                    "best_epoch_mode": mode,
                    "best_epoch_min": int(epoch_values.min()),
                    "best_epoch_max": int(epoch_values.max()),
                    "best_epoch_values": json.dumps(
                        sorted(int(value) for value in epoch_values),
                        separators=(",", ":"),
                    ),
                }
            )
        else:
            row.update(
                {
                    "best_epoch_mean": float("nan"),
                    "best_epoch_mode": float("nan"),
                    "best_epoch_min": float("nan"),
                    "best_epoch_max": float("nan"),
                    "best_epoch_values": "[]",
                }
            )
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(
        [CONFIGURATION_NAME_COLUMN, CONFIGURATION_FINGERPRINT_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def pareto_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    """Compute domination counts for fully valid, floor-passing candidates.

    Mean precision and local privacy are maximized; mean execution time is
    minimized.  Mean selected-parameter fraction is also minimized when it is
    finite for every eligible candidate.
    """

    required = {
        "run_count",
        "valid_count",
        "utility_floor_pass_count",
        "precision_at_10_mean",
        "local_privacy_proxy_mean",
        "execution_time_seconds_mean",
        "selected_parameter_fraction_mean",
        CONFIGURATION_NAME_COLUMN,
        CONFIGURATION_FINGERPRINT_COLUMN,
    }
    missing = required - set(summary.columns)
    if missing:
        raise KeyError(f"Colonne Pareto mancanti: {sorted(missing)}")

    eligible = summary.loc[
        (summary["run_count"] > 0)
        & (summary["valid_count"] == summary["run_count"])
        & (summary["utility_floor_pass_count"] == summary["run_count"])
    ].copy()
    if eligible.empty:
        eligible["selected_parameter_fraction_objective_used"] = pd.Series(
            dtype=bool
        )
        eligible["pareto_dominated_count"] = pd.Series(dtype=np.int64)
        eligible["is_pareto_optimal"] = pd.Series(dtype=bool)
        return eligible

    mandatory_objectives = [
        ("precision_at_10_mean", "max"),
        ("local_privacy_proxy_mean", "max"),
        ("execution_time_seconds_mean", "min"),
    ]
    for column, _direction in mandatory_objectives:
        numeric = pd.to_numeric(eligible[column], errors="coerce")
        if not bool(np.isfinite(numeric.to_numpy(dtype=np.float64)).all()):
            raise ValueError(f"Valori Pareto mancanti o non finiti in {column}.")
        eligible[column] = numeric

    selected = pd.to_numeric(
        eligible["selected_parameter_fraction_mean"], errors="coerce"
    )
    use_selected_fraction = bool(
        np.isfinite(selected.to_numpy(dtype=np.float64)).all()
    )
    objectives = list(mandatory_objectives)
    if use_selected_fraction:
        eligible["selected_parameter_fraction_mean"] = selected
        objectives.append(("selected_parameter_fraction_mean", "min"))

    values = eligible[[column for column, _direction in objectives]].to_numpy(
        dtype=np.float64
    )
    minimize_values = values.copy()
    for index, (_column, direction) in enumerate(objectives):
        if direction == "max":
            minimize_values[:, index] *= -1.0

    dominated_counts: list[int] = []
    for index, current in enumerate(minimize_values):
        dominated_count = 0
        for other_index, other in enumerate(minimize_values):
            if index == other_index:
                continue
            if bool(np.all(other <= current) and np.any(other < current)):
                dominated_count += 1
        dominated_counts.append(dominated_count)

    eligible["selected_parameter_fraction_objective_used"] = use_selected_fraction
    eligible["pareto_dominated_count"] = dominated_counts
    eligible["is_pareto_optimal"] = eligible["pareto_dominated_count"] == 0
    return eligible.sort_values(
        [
            "is_pareto_optimal",
            "pareto_dominated_count",
            CONFIGURATION_NAME_COLUMN,
            CONFIGURATION_FINGERPRINT_COLUMN,
        ],
        ascending=[False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def aggregate_all_candidates(
    input_dir: str | Path,
    *,
    expected_seeds: Sequence[int] | None = None,
    expected_seed_count: int | None = None,
) -> AggregationResult:
    """Run the complete deterministic aggregation without writing outputs."""

    root = Path(input_dir)
    raw, sources = load_all_candidate_results(root, require_full_evidence=True)
    discovered_seeds = sorted(source.seed for source in sources)
    resolved_expected, resolved_expected_count = _validated_expected_seeds(
        discovered_seeds,
        expected_seeds=expected_seeds,
        expected_seed_count=expected_seed_count,
    )
    summary = summarize_candidate_results(
        raw,
        expected_seeds=resolved_expected,
        expected_seed_count=resolved_expected_count,
    )
    pareto = pareto_analysis(summary)

    missing_seed_directories = (
        sorted(set(resolved_expected) - set(discovered_seeds))
        if resolved_expected is not None
        else []
    )
    unexpected_seed_directories = (
        sorted(set(discovered_seeds) - set(resolved_expected))
        if resolved_expected is not None
        else []
    )
    scope_counts = {
        str(scope): int(count)
        for scope, count in raw[CONFIGURATION_SCOPE_COLUMN]
        .value_counts(sort=False)
        .sort_index()
        .items()
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "input_dir": str(root),
        "source_files": [
            source.comparison_path.relative_to(root).as_posix() for source in sources
        ],
        "effective_config_files": [
            source.effective_config_path.relative_to(root).as_posix()
            for source in sources
            if source.effective_config_path is not None
        ],
        "search_metadata_files": [
            source.search_metadata_path.relative_to(root).as_posix()
            for source in sources
            if source.search_metadata_path is not None
        ],
        "evidence_mode": "full",
        "discovered_seeds": discovered_seeds,
        "expected_seeds": resolved_expected,
        "expected_seed_count": resolved_expected_count,
        "missing_seed_directories": missing_seed_directories,
        "unexpected_seed_directories": unexpected_seed_directories,
        "raw_run_count": int(len(raw)),
        "valid_run_count": int(raw["valid"].sum()),
        "invalid_run_count": int((~raw["valid"]).sum()),
        "configuration_count": int(len(summary)),
        "pareto_eligible_count": int(len(pareto)),
        "pareto_optimal_count": int(pareto.get("is_pareto_optimal", pd.Series(dtype=bool)).sum()),
        "fingerprint_algorithm": "sha256-canonical-json-v1",
        "fingerprint_semantic_fields": list(SEMANTIC_CONFIGURATION_FIELDS),
        "fingerprint_scope_counts": scope_counts,
        "inactive_field_normalization": {
            "gradient_ascent_steps=0": [
                "gradient_ascent_learning_rate",
                "gradient_ascent_batch_size",
                "gradient_ascent_retain_distillation_weight",
            ],
            "recalibrate_batchnorm=false": [
                "batchnorm_recalibration_batch_size"
            ],
        },
        "metric_aggregation": "valid runs only",
        "standard_deviation": "population (ddof=0)",
        "pareto_objectives": {
            "maximize": ["precision_at_10_mean", "local_privacy_proxy_mean"],
            "minimize": [
                "execution_time_seconds_mean",
                "selected_parameter_fraction_mean when available for every candidate",
            ],
        },
        "privacy_proxy_limitation": LOCAL_PRIVACY_DISCLAIMER,
    }
    return AggregationResult(raw=raw, summary=summary, pareto=pareto, metadata=metadata)


def write_aggregation_outputs(
    result: AggregationResult,
    *,
    raw_path: str | Path,
    summary_path: str | Path,
    pareto_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Path]:
    """Write all aggregation artifacts and return their resolved destinations."""

    destinations = {
        "raw": Path(raw_path),
        "summary": Path(summary_path),
        "pareto": Path(pareto_path),
        "metadata": Path(metadata_path),
    }
    resolved_destinations = [path.resolve() for path in destinations.values()]
    if len(resolved_destinations) != len(set(resolved_destinations)):
        raise ValueError(
            "Le destinazioni raw, summary, Pareto e metadata devono essere distinte."
        )
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.raw.to_csv(destinations["raw"], index=False)
    result.summary.to_csv(destinations["summary"], index=False)
    result.pareto.to_csv(destinations["pareto"], index=False)
    metadata = {
        **result.metadata,
        "outputs": {name: str(path) for name, path in destinations.items()},
    }
    destinations["metadata"].write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destinations


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def validate_aggregation_output_paths(
    *,
    input_dir: str | Path,
    sources: Sequence[SearchResultFile],
    repository_root: str | Path,
    raw_path: str | Path,
    summary_path: str | Path,
    pareto_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Path]:
    """Reject protected, source, and colliding destinations before any write."""

    destinations = {
        "raw": Path(raw_path),
        "summary": Path(summary_path),
        "pareto": Path(pareto_path),
        "metadata": Path(metadata_path),
    }
    resolved = {name: path.resolve() for name, path in destinations.items()}
    if len(set(resolved.values())) != len(resolved):
        raise ValueError(
            "Le destinazioni raw, summary, Pareto e metadata devono essere distinte."
        )

    repository = Path(repository_root).resolve()
    canonical = (repository / "configs" / "final_config.json").resolve()
    forbidden_directories = (
        (repository / "outputs" / "final_run").resolve(),
        (repository / "submission").resolve(),
    )
    root = Path(input_dir).resolve()
    protected_sources: set[Path] = set()
    for source in sources:
        protected_sources.add(source.comparison_path.resolve())
        if source.effective_config_path is not None:
            protected_sources.add(source.effective_config_path.resolve())
        if source.search_metadata_path is not None:
            protected_sources.add(source.search_metadata_path.resolve())
    for root_source_name in ("effective_search_config.json", "search_metadata.json"):
        root_source = root / root_source_name
        if root_source.is_file():
            protected_sources.add(root_source.resolve())

    for name, destination in resolved.items():
        if destination == canonical:
            raise ValueError(
                f"La destinazione {name} non puo' sovrascrivere "
                "configs/final_config.json."
            )
        for forbidden in forbidden_directories:
            if _is_within(destination, forbidden):
                raise ValueError(
                    f"La destinazione {name} non puo' trovarsi nel percorso "
                    f"protetto {forbidden}."
                )
        if destination in protected_sources:
            raise ValueError(
                f"La destinazione {name} coincide con un file di evidenza sorgente: "
                f"{destination}."
            )
    return destinations
