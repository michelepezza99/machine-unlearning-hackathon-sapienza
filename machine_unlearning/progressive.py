"""Pure, deterministic generators for the progressive search configurations."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from .search import validate_search_config
from .search_aggregation import canonical_configuration_json


RankingDirection = Literal["asc", "desc"]
RankingSpecification = tuple[tuple[str, RankingDirection], ...]

STAGE2_QUANTILES: Final = (0.40, 0.50, 0.60)
STAGE2_BATCHNORM_CHOICES: Final = (False, True)

REFERENCE_REPAIR_LEARNING_RATE: Final = 0.00013558725141944915
APPROXIMATE_GRADIENT_ASCENT_LEARNING_RATE: Final = (
    REFERENCE_REPAIR_LEARNING_RATE * 0.1
)
NO_GRADIENT_ASCENT_LEARNING_RATE: Final = 0.000001

STAGE2_RANKING: Final[RankingSpecification] = (
    ("local_search_score", "desc"),
    ("local_privacy_proxy", "desc"),
    ("precision_at_10", "desc"),
    ("execution_time_seconds", "asc"),
)

STAGE3_RANKING: Final[RankingSpecification] = (
    ("local_privacy_proxy", "desc"),
    ("precision_at_10", "desc"),
    ("execution_time_seconds", "asc"),
    ("local_search_score", "desc"),
)

STAGE4_RANKING: Final[RankingSpecification] = (
    ("local_privacy_proxy", "desc"),
    ("precision_at_10", "desc"),
    ("utility_ratio", "desc"),
    ("execution_time_seconds", "asc"),
    ("selected_parameter_fraction", "asc"),
    ("local_search_score", "desc"),
)

GRADIENT_ASCENT_PROFILES: Final = (
    ("ga0", 0, NO_GRADIENT_ASCENT_LEARNING_RATE, "senza gradient ascent"),
    ("ga2_lr5em6", 2, 5e-6, "2 step di gradient ascent a learning rate 5e-6"),
    ("ga2_lr1em5", 2, 1e-5, "2 step di gradient ascent a learning rate 1e-5"),
    ("ga4_lr1em5", 4, 1e-5, "4 step di gradient ascent a learning rate 1e-5"),
    (
        "ga4_lr1p35em5",
        4,
        APPROXIMATE_GRADIENT_ASCENT_LEARNING_RATE,
        "4 step di gradient ascent al learning rate derivato dal repair di riferimento",
    ),
    ("ga8_lr5em6", 8, 5e-6, "8 step di gradient ascent a learning rate 5e-6"),
)

REPAIR_PROFILES: Final = (
    (
        "conservative",
        5e-5,
        1.0,
        1e-4,
        "repair conservativo con forte distillazione",
    ),
    (
        "flexible",
        REFERENCE_REPAIR_LEARNING_RATE,
        0.25,
        1e-5,
        "repair piu' flessibile con regolarizzazione ridotta",
    ),
    (
        "strong",
        2.5e-4,
        0.5,
        1e-4,
        "repair forte con learning rate maggiore",
    ),
    (
        "regularized",
        REFERENCE_REPAIR_LEARNING_RATE,
        0.5,
        1e-3,
        "repair fortemente regolarizzato",
    ),
)

STRUCTURAL_STAGE2_FIELDS: Final = (
    "top_fraction",
    "minimum_dampening_factor",
)

CANDIDATE_FLOAT_FIELDS: Final = (
    "top_fraction",
    "forget_absolute_quantile",
    "minimum_dampening_factor",
    "dampening_strength",
    "fisher_ratio_power",
    "repair_learning_rate",
    "repair_weight_decay",
    "supervised_loss_weight",
    "distillation_weight",
    "parameter_regularization_weight",
    "selected_parameter_weight",
    "gradient_clip",
    "gradient_ascent_learning_rate",
    "gradient_ascent_retain_distillation_weight",
)
CANDIDATE_INTEGER_FIELDS: Final = (
    "repair_batch_size",
    "repair_max_epochs",
    "repair_patience",
    "gradient_ascent_steps",
    "gradient_ascent_batch_size",
    "batchnorm_recalibration_batch_size",
)
CANDIDATE_BOOLEAN_FIELDS: Final = (
    "freeze_selected_during_repair",
    "recalibrate_batchnorm",
)

_TRUE_VALUES: Final = frozenset({"1", "true", "t", "yes", "y"})
_FALSE_VALUES: Final = frozenset({"0", "false", "f", "no", "n"})
_NAME_COLUMNS: Final = ("name", "configuration", "configuration_name")


def normalize_boolean(value: object, *, field_name: str) -> bool:
    """Normalize common CSV Boolean encodings or fail with an actionable error."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in (0.0, 1.0):
            return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    raise ValueError(f"{field_name} contiene un booleano non riconosciuto: {value!r}.")


def semantic_candidate_fingerprint(
    common_candidate: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    """Return the canonical semantic identity of an effective candidate.

    The aggregation module owns the field typing and inactive-knob policy used
    across the repository.  Reusing it here makes ``1`` and ``1.0`` equivalent,
    and ignores GA/BatchNorm settings when their corresponding phase is off.
    """
    effective = {**deepcopy(dict(common_candidate)), **deepcopy(dict(candidate))}
    return canonical_configuration_json(effective)


def deduplicate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    common_candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Keep the first candidate for each semantic configuration."""
    unique: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"candidates[{index}] deve essere un mapping.")
        fingerprint = semantic_candidate_fingerprint(common_candidate, candidate)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        unique.append(deepcopy(dict(candidate)))
    return unique


def _validated_template(template_config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(template_config, Mapping):
        raise TypeError("La configurazione template deve essere un mapping.")
    copied = deepcopy(dict(template_config))
    validate_search_config(copied)
    return copied


def _name_column(results: pd.DataFrame) -> str:
    for column in _NAME_COLUMNS:
        if column in results.columns:
            return column
    raise KeyError(
        "I risultati devono contenere una colonna nome tra "
        f"{list(_NAME_COLUMNS)}."
    )


def _prepare_evidence(
    results: pd.DataFrame,
    *,
    numeric_fields: Sequence[str],
    boolean_fields: Sequence[str] = (),
) -> pd.DataFrame:
    if not isinstance(results, pd.DataFrame):
        raise TypeError("I risultati devono essere un DataFrame pandas.")
    if results.empty:
        raise ValueError("Il file dei risultati non contiene candidati.")
    name_column = _name_column(results)
    required = {"valid", "utility_floor_pass", *numeric_fields, *boolean_fields}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Colonne richieste mancanti dai risultati: {sorted(missing)}")

    prepared = results.copy()
    if prepared[name_column].isna().any():
        raise ValueError("I risultati contengono un nome configurazione mancante.")
    prepared["_configuration_name"] = prepared[name_column].map(str).str.strip()
    if (prepared["_configuration_name"] == "").any():
        raise ValueError("I risultati contengono un nome configurazione vuoto.")
    duplicates = sorted(
        prepared.loc[
            prepared["_configuration_name"].duplicated(keep=False),
            "_configuration_name",
        ].unique()
    )
    if duplicates:
        raise ValueError(f"Risultati duplicati per configurazione: {duplicates}")

    for flag in ("valid", "utility_floor_pass"):
        prepared[flag] = [
            normalize_boolean(value, field_name=flag) for value in prepared[flag]
        ]
    prepared = prepared.loc[
        prepared["valid"] & prepared["utility_floor_pass"]
    ].copy()
    if prepared.empty:
        raise ValueError(
            "Nessun candidato valido supera l'utility floor; la generazione si arresta."
        )

    for field in numeric_fields:
        prepared[field] = pd.to_numeric(prepared[field], errors="coerce")
        invalid = ~np.isfinite(prepared[field].to_numpy(dtype=np.float64))
        if bool(invalid.any()):
            names = prepared.loc[invalid, "_configuration_name"].tolist()
            raise ValueError(
                f"La colonna {field!r} non e' finita per i candidati: {names}."
            )
    for field in boolean_fields:
        prepared[field] = [
            normalize_boolean(value, field_name=field) for value in prepared[field]
        ]
    return prepared.reset_index(drop=True)


def _rank_evidence(
    evidence: pd.DataFrame, ranking: RankingSpecification
) -> pd.DataFrame:
    columns = [field for field, _ in ranking] + ["_configuration_name"]
    ascending = [direction == "asc" for _, direction in ranking] + [True]
    return evidence.sort_values(
        columns,
        ascending=ascending,
        kind="mergesort",
    ).reset_index(drop=True)


def _metric_snapshot(row: pd.Series, ranking: RankingSpecification) -> dict[str, float]:
    return {field: float(row[field]) for field, _ in ranking}


def select_stage2_families(
    results: pd.DataFrame, *, family_count: int = 3
) -> list[dict[str, Any]]:
    """Select distinct ``(top_fraction, dampening)`` families from Stage 1."""
    if family_count <= 0:
        raise ValueError("family_count deve essere positivo.")
    numeric_fields = (
        *STRUCTURAL_STAGE2_FIELDS,
        *(field for field, _ in STAGE2_RANKING),
    )
    ranked = _rank_evidence(
        _prepare_evidence(results, numeric_fields=numeric_fields), STAGE2_RANKING
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, float]] = set()
    for _, row in ranked.iterrows():
        family = (
            float(row["top_fraction"]),
            float(row["minimum_dampening_factor"]),
        )
        if family in seen:
            continue
        seen.add(family)
        rank = len(selected) + 1
        selected.append(
            {
                "rank": rank,
                "top_fraction": family[0],
                "minimum_dampening_factor": family[1],
                "representative_configuration": str(row["_configuration_name"]),
                "evidence": _metric_snapshot(row, STAGE2_RANKING),
                "selection_reason": (
                    f"Famiglia strutturale distinta al rango {rank}; il rappresentante "
                    "e' il migliore della famiglia secondo score locale, privacy, "
                    "Precision@10, tempo e nome come tie-break deterministico."
                ),
            }
        )
        if len(selected) == family_count:
            break
    if len(selected) < family_count:
        raise ValueError(
            f"Servono {family_count} famiglie strutturali valide distinte; "
            f"ne sono disponibili {len(selected)}."
        )
    return selected


def select_stage3_structures(
    results: pd.DataFrame, *, structure_count: int = 2
) -> list[dict[str, Any]]:
    """Select privacy-first distinct Stage 2 structures for bounded refinement."""
    if structure_count <= 0:
        raise ValueError("structure_count deve essere positivo.")
    numeric_fields = (
        "top_fraction",
        "minimum_dampening_factor",
        "forget_absolute_quantile",
        *(field for field, _ in STAGE3_RANKING),
    )
    ranked = _rank_evidence(
        _prepare_evidence(
            results,
            numeric_fields=numeric_fields,
            boolean_fields=("recalibrate_batchnorm",),
        ),
        STAGE3_RANKING,
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, bool]] = set()
    for _, row in ranked.iterrows():
        structure = (
            float(row["top_fraction"]),
            float(row["minimum_dampening_factor"]),
            float(row["forget_absolute_quantile"]),
            bool(row["recalibrate_batchnorm"]),
        )
        if structure in seen:
            continue
        seen.add(structure)
        rank = len(selected) + 1
        selected.append(
            {
                "rank": rank,
                "top_fraction": structure[0],
                "minimum_dampening_factor": structure[1],
                "forget_absolute_quantile": structure[2],
                "recalibrate_batchnorm": structure[3],
                "representative_configuration": str(row["_configuration_name"]),
                "evidence": _metric_snapshot(row, STAGE3_RANKING),
                "selection_reason": (
                    f"Struttura distinta al rango {rank}; priorita' gerarchica a "
                    "privacy locale, Precision@10 e tempo, con score e nome come "
                    "tie-break deterministici."
                ),
            }
        )
        if len(selected) == structure_count:
            break
    if len(selected) < structure_count:
        raise ValueError(
            f"Servono {structure_count} strutture valide distinte; "
            f"ne sono disponibili {len(selected)}."
        )
    return selected


def _decimal_token(value: float, *, scale: int, minimum_integer_digits: int) -> str:
    scaled = Decimal(str(value)) * Decimal(scale)
    text = format(scaled.normalize(), "f")
    if "." in text:
        integer, fraction = text.split(".", maxsplit=1)
        fraction = fraction.rstrip("0")
    else:
        integer, fraction = text, ""
    sign = "m" if integer.startswith("-") else ""
    integer = integer.lstrip("-").zfill(minimum_integer_digits)
    return sign + integer + (f"p{fraction}" if fraction else "")


def structural_name(
    *,
    top_fraction: float,
    minimum_dampening_factor: float,
    forget_absolute_quantile: float | None = None,
    recalibrate_batchnorm: bool | None = None,
) -> str:
    """Build a stable, self-describing structural candidate name."""
    parts = [
        "tf_" + _decimal_token(top_fraction, scale=100, minimum_integer_digits=1),
        "d" + _decimal_token(
            minimum_dampening_factor, scale=100, minimum_integer_digits=3
        ),
    ]
    if forget_absolute_quantile is not None:
        parts.append(
            "q"
            + _decimal_token(
                forget_absolute_quantile, scale=100, minimum_integer_digits=3
            )
        )
    if recalibrate_batchnorm is not None:
        parts.append("with_bn" if recalibrate_batchnorm else "no_bn")
    return "_".join(parts)


def _generation_metadata(
    *,
    stage: int,
    source_results: str | None,
    candidate_count: int,
    automatic_gradient_ascent_variants: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": stage,
        "source_results": source_results,
        "candidate_count": candidate_count,
        "automatic_gradient_ascent_variants": automatic_gradient_ascent_variants,
        "privacy_metric_limitation": (
            "La proxy locale rispetto al retraining non equivale alla MIA "
            "ufficiale nascosta."
        ),
    }


def generate_stage2_config(
    template_config: Mapping[str, Any],
    results: pd.DataFrame,
    *,
    source_results: str | None = None,
) -> dict[str, Any]:
    """Generate 18 quantile/BatchNorm candidates from three Stage 1 families."""
    generated = _validated_template(template_config)
    families = select_stage2_families(results)
    candidates: list[dict[str, Any]] = []
    rationales: list[dict[str, str]] = []
    for family in families:
        for quantile in STAGE2_QUANTILES:
            for recalibrate in STAGE2_BATCHNORM_CHOICES:
                name = structural_name(
                    top_fraction=float(family["top_fraction"]),
                    minimum_dampening_factor=float(
                        family["minimum_dampening_factor"]
                    ),
                    forget_absolute_quantile=quantile,
                    recalibrate_batchnorm=recalibrate,
                )
                candidates.append(
                    {
                        "name": name,
                        "top_fraction": float(family["top_fraction"]),
                        "minimum_dampening_factor": float(
                            family["minimum_dampening_factor"]
                        ),
                        "forget_absolute_quantile": quantile,
                        "recalibrate_batchnorm": recalibrate,
                    }
                )
                rationales.append(
                    {
                        "name": name,
                        "reason": (
                            f"Raffina la famiglia Stage 1 di rango {family['rank']} "
                            f"variando quantile={quantile:.2f} e ricalibrazione "
                            f"BatchNorm={recalibrate}."
                        ),
                    }
                )
    candidates = deduplicate_candidates(
        candidates, common_candidate=generated["common_candidate"]
    )
    if len(candidates) != 18:
        raise RuntimeError(
            f"Stage 2 deve produrre 18 candidati, ottenuti {len(candidates)}."
        )
    generated["add_gradient_ascent_variants"] = 6
    generated["candidates"] = candidates
    metadata = _generation_metadata(
        stage=2,
        source_results=source_results,
        candidate_count=len(candidates),
        automatic_gradient_ascent_variants=6,
    )
    metadata.update(
        {
            "selection_ranking": list(STAGE2_RANKING),
            "selected_families": families,
            "candidate_rationales": rationales,
        }
    )
    generated["progressive_generation"] = metadata
    validate_search_config(generated)
    return generated


def generate_stage3_config(
    template_config: Mapping[str, Any],
    results: pd.DataFrame,
    *,
    source_results: str | None = None,
) -> dict[str, Any]:
    """Generate a bounded 20-candidate GA/repair design from Stage 2 evidence."""
    generated = _validated_template(template_config)
    structures = select_stage3_structures(results)
    candidates: list[dict[str, Any]] = []
    rationales: list[dict[str, str]] = []
    for structure in structures:
        base_name = structural_name(
            top_fraction=float(structure["top_fraction"]),
            minimum_dampening_factor=float(
                structure["minimum_dampening_factor"]
            ),
            forget_absolute_quantile=float(structure["forget_absolute_quantile"]),
            recalibrate_batchnorm=bool(structure["recalibrate_batchnorm"]),
        )
        structural_fields = {
            "top_fraction": float(structure["top_fraction"]),
            "minimum_dampening_factor": float(
                structure["minimum_dampening_factor"]
            ),
            "forget_absolute_quantile": float(
                structure["forget_absolute_quantile"]
            ),
            "recalibrate_batchnorm": bool(structure["recalibrate_batchnorm"]),
        }
        for profile_name, steps, learning_rate, description in GRADIENT_ASCENT_PROFILES:
            name = f"{base_name}__repair_reference__{profile_name}"
            candidates.append(
                {
                    "name": name,
                    **structural_fields,
                    "gradient_ascent_steps": steps,
                    "gradient_ascent_learning_rate": learning_rate,
                    "repair_learning_rate": REFERENCE_REPAIR_LEARNING_RATE,
                    "distillation_weight": 0.5,
                    "parameter_regularization_weight": 1e-4,
                }
            )
            rationales.append(
                {
                    "name": name,
                    "reason": (
                        f"Struttura Stage 2 di rango {structure['rank']}; repair di "
                        f"riferimento e {description}."
                    ),
                }
            )
        for profile_name, learning_rate, distillation, regularization, description in (
            REPAIR_PROFILES
        ):
            name = f"{base_name}__repair_{profile_name}__ga0"
            candidates.append(
                {
                    "name": name,
                    **structural_fields,
                    "gradient_ascent_steps": 0,
                    "gradient_ascent_learning_rate": NO_GRADIENT_ASCENT_LEARNING_RATE,
                    "repair_learning_rate": learning_rate,
                    "distillation_weight": distillation,
                    "parameter_regularization_weight": regularization,
                }
            )
            rationales.append(
                {
                    "name": name,
                    "reason": (
                        f"Struttura Stage 2 di rango {structure['rank']}; "
                        f"{description} "
                        "senza gradient ascent per isolare l'effetto del repair."
                    ),
                }
            )
    candidates = deduplicate_candidates(
        candidates, common_candidate=generated["common_candidate"]
    )
    expected = 10 * len(structures)
    if len(candidates) != expected or expected > 20:
        raise RuntimeError(
            f"Stage 3 attende {expected} candidati unici (massimo 20), "
            f"ottenuti {len(candidates)}."
        )
    generated["add_gradient_ascent_variants"] = 0
    generated["candidates"] = candidates
    metadata = _generation_metadata(
        stage=3,
        source_results=source_results,
        candidate_count=len(candidates),
        automatic_gradient_ascent_variants=0,
    )
    metadata.update(
        {
            "selection_ranking": list(STAGE3_RANKING),
            "selected_structures": structures,
            "design": (
                "Per struttura: sei profili GA col repair di riferimento e quattro "
                "repair alternativi senza GA; nessun prodotto cartesiano completo."
            ),
            "candidate_rationales": rationales,
        }
    )
    generated["progressive_generation"] = metadata
    validate_search_config(generated)
    return generated


def _candidate_from_evidence(row: pd.Series) -> dict[str, Any]:
    candidate: dict[str, Any] = {"name": str(row["_configuration_name"])}
    for field in CANDIDATE_FLOAT_FIELDS:
        candidate[field] = float(row[field])
    for field in CANDIDATE_INTEGER_FIELDS:
        value = float(row[field])
        if not value.is_integer():
            raise ValueError(
                f"{field} deve essere intero per {row['_configuration_name']!r}."
            )
        candidate[field] = int(value)
    for field in CANDIDATE_BOOLEAN_FIELDS:
        candidate[field] = bool(row[field])
    return candidate


def _stage4_signature(record: Mapping[str, Any]) -> tuple[Any, ...]:
    candidate = record["effective_candidate"]
    return (
        float(candidate["top_fraction"]),
        float(candidate["minimum_dampening_factor"]),
        float(candidate["forget_absolute_quantile"]),
        bool(candidate["recalibrate_batchnorm"]),
    )


def _ga_signature(record: Mapping[str, Any]) -> tuple[int, float]:
    candidate = record["effective_candidate"]
    return (
        int(candidate["gradient_ascent_steps"]),
        float(candidate["gradient_ascent_learning_rate"]),
    )


def _repair_signature(record: Mapping[str, Any]) -> tuple[float, float, float]:
    candidate = record["effective_candidate"]
    return (
        float(candidate["repair_learning_rate"]),
        float(candidate["distillation_weight"]),
        float(candidate["parameter_regularization_weight"]),
    )


def select_stage4_finalists(
    results: pd.DataFrame, *, finalist_count: int = 4
) -> list[dict[str, Any]]:
    """Select ranked Stage 3 finalists while balancing structural/profile diversity."""
    if finalist_count <= 0:
        raise ValueError("finalist_count deve essere positivo.")
    numeric_fields = tuple(
        dict.fromkeys(
            (
                *CANDIDATE_FLOAT_FIELDS,
                *CANDIDATE_INTEGER_FIELDS,
                *(field for field, _ in STAGE4_RANKING),
            )
        )
    )
    ranked = _rank_evidence(
        _prepare_evidence(
            results,
            numeric_fields=numeric_fields,
            boolean_fields=CANDIDATE_BOOLEAN_FIELDS,
        ),
        STAGE4_RANKING,
    )
    records: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for rank_index, (_, row) in enumerate(ranked.iterrows(), start=1):
        candidate = _candidate_from_evidence(row)
        fingerprint = semantic_candidate_fingerprint({}, candidate)
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        records.append(
            {
                "name": candidate["name"],
                "evidence_rank": rank_index,
                "evidence": _metric_snapshot(row, STAGE4_RANKING),
                "effective_candidate": candidate,
            }
        )
    if len(records) < finalist_count:
        raise ValueError(
            f"Servono {finalist_count} configurazioni valide uniche; "
            f"ne sono disponibili {len(records)}."
        )

    selected: list[dict[str, Any]] = []
    remaining = list(records)
    best_by_structure: list[dict[str, Any]] = []
    seen_structures: set[tuple[Any, ...]] = set()
    for record in records:
        signature = _stage4_signature(record)
        if signature not in seen_structures:
            seen_structures.add(signature)
            best_by_structure.append(record)
    for record in best_by_structure[:finalist_count]:
        chosen = deepcopy(record)
        chosen["selection_reason"] = (
            "Miglior candidato della propria struttura secondo il ranking Stage 4."
        )
        selected.append(chosen)
        remaining.remove(record)

    while len(selected) < finalist_count:
        structure_counts = Counter(_stage4_signature(item) for item in selected)
        used_ga = {_ga_signature(item) for item in selected}
        used_repair = {_repair_signature(item) for item in selected}
        chosen = min(
            remaining,
            key=lambda item: (
                structure_counts[_stage4_signature(item)],
                0 if _ga_signature(item) not in used_ga else 1,
                0 if _repair_signature(item) not in used_repair else 1,
                int(item["evidence_rank"]),
                str(item["name"]),
            ),
        )
        remaining.remove(chosen)
        selected_record = deepcopy(chosen)
        selected_record["selection_reason"] = (
            "Riempimento deterministico che bilancia il numero per struttura, "
            "poi preferisce profili GA e repair non ancora rappresentati."
        )
        selected.append(selected_record)
    return selected


def generate_stage4_config(
    template_config: Mapping[str, Any],
    results: pd.DataFrame,
    *,
    source_results: str | None = None,
    finalist_count: int = 4,
) -> dict[str, Any]:
    """Generate a Stage 4 multi-seed search config from Stage 3 evidence."""
    generated = _validated_template(template_config)
    finalists = select_stage4_finalists(results, finalist_count=finalist_count)
    candidates = [item["effective_candidate"] for item in finalists]
    candidates = deduplicate_candidates(
        candidates, common_candidate=generated["common_candidate"]
    )
    if len(candidates) != finalist_count:
        raise RuntimeError(
            "La deduplicazione dei finalisti Stage 4 ha ridotto il budget sotto "
            f"{finalist_count}."
        )
    generated["add_gradient_ascent_variants"] = 0
    generated["candidates"] = candidates
    metadata = _generation_metadata(
        stage=4,
        source_results=source_results,
        candidate_count=len(candidates),
        automatic_gradient_ascent_variants=0,
    )
    metadata.update(
        {
            "selection_ranking": list(STAGE4_RANKING),
            "diversity_policy": (
                "Prima il miglior candidato per struttura; poi bilanciamento delle "
                "strutture e novita' dei profili GA/repair, con ranking e nome come "
                "tie-break."
            ),
            "selected_finalists": [
                {
                    key: value
                    for key, value in item.items()
                    if key != "effective_candidate"
                }
                for item in finalists
            ],
        }
    )
    generated["progressive_generation"] = metadata
    validate_search_config(generated)
    return generated
