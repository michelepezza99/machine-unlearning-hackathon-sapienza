"""Synthetic tests for deterministic all-candidate search aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machine_unlearning.search_aggregation import (
    CONFIGURATION_FINGERPRINT_COLUMN,
    CONFIGURATION_NAME_COLUMN,
    aggregate_all_candidates,
    configuration_fingerprint,
    discover_search_result_files,
    infer_expected_seeds_from_root_config,
    normalize_boolean,
    pareto_analysis,
    validate_aggregation_output_paths,
)
from scripts import summarize_all_candidates


def _candidate_row(
    seed: int,
    name: str,
    *,
    top_fraction: float = 0.01,
    valid: bool | str = True,
    utility_floor_pass: bool | str = True,
    precision_at_10: float = 0.8,
    best_epoch: int = 2,
) -> dict[str, object]:
    is_valid = str(valid).strip().lower() != "false" if isinstance(valid, str) else valid
    return {
        "config_index": 1,
        "seed": seed,
        "name": name,
        "top_fraction": top_fraction,
        "forget_absolute_quantile": 0.5,
        "minimum_dampening_factor": 0.9,
        "dampening_strength": 1.0,
        "fisher_ratio_power": 1.0,
        "gradient_ascent_steps": 0,
        "gradient_ascent_learning_rate": 1e-5,
        "gradient_ascent_batch_size": 8,
        "gradient_ascent_retain_distillation_weight": 1.0,
        "repair_learning_rate": 1e-4,
        "repair_weight_decay": 0.0,
        "repair_batch_size": 8,
        "repair_max_epochs": 3,
        "repair_patience": 1,
        "supervised_loss_weight": 1.0,
        "distillation_weight": 0.5,
        "parameter_regularization_weight": 1e-4,
        "selected_parameter_weight": 1.0,
        "gradient_clip": 1.0,
        "freeze_selected_during_repair": "True",
        "recalibrate_batchnorm": "False",
        "batchnorm_recalibration_batch_size": 16,
        "utility_floor_ratio": 0.9,
        "status": "success" if is_valid else "failed",
        "valid": valid,
        "utility_floor_pass": utility_floor_pass,
        "error_type": None if is_valid else "RuntimeError",
        "error_message": None if is_valid else "synthetic failure",
        "selected_parameter_count": 10 if is_valid else np.nan,
        "selected_parameter_fraction": 0.1 if is_valid else np.nan,
        "gradient_ascent_used": "False",
        "batchnorm_recalibration_used": False,
        "precision_at_10": precision_at_10 if is_valid else np.nan,
        "utility_ratio": precision_at_10 / 0.8 if is_valid else np.nan,
        "validation_bce": 0.4 if is_valid else np.nan,
        "forget_bce": 0.6 if is_valid else np.nan,
        "local_privacy_proxy": 0.7 if is_valid else np.nan,
        "execution_time_seconds": 2.0 if is_valid else np.nan,
        "local_search_score": 0.75 if is_valid else np.nan,
        "best_epoch": best_epoch if is_valid else -1,
    }


def _effective_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed": 92,
        "validation_fraction": 0.2,
        "evaluation_batch_size": 16,
        "utility_floor_ratio": 0.9,
        "fisher": {
            "teacher_batch_size": 16,
            "fisher_retain_sample_size": 32,
            "fisher_forget_sample_size": 24,
            "fisher_batch_size": 4,
            "include_bias": False,
            "include_batchnorm_affine": False,
        },
    }


def _write_seed(
    root: Path,
    seed: int,
    rows: list[dict[str, object]],
    *,
    effective_config: bool = True,
    mode: str = "full",
    status: str = "completed",
) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(seed_dir / "search_comparison.csv", index=False)
    if effective_config:
        config = _effective_config()
        config["seed"] = seed
        (seed_dir / "effective_search_config.json").write_text(
            json.dumps(config), encoding="utf-8"
        )
    (seed_dir / "search_metadata.json").write_text(
        json.dumps({"status": status, "mode": mode, "seed": seed}),
        encoding="utf-8",
    )


def _write_root_effective_config(root: Path, seeds: list[int]) -> None:
    (root / "effective_search_config.json").write_text(
        json.dumps({"mode": "full", "seeds": seeds, "config": _effective_config()}),
        encoding="utf-8",
    )


def test_discovery_is_deterministic_and_empty_input_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"seed_\*/search_comparison"):
        discover_search_result_files(tmp_path)
    _write_seed(tmp_path, 10, [_candidate_row(10, "candidate")])
    _write_seed(tmp_path, 2, [_candidate_row(2, "candidate")])
    discovered = discover_search_result_files(tmp_path)
    assert [item.seed for item in discovered] == [2, 10]


def test_boolean_normalization_is_strict() -> None:
    assert normalize_boolean(True)
    assert normalize_boolean(" TRUE ")
    assert not normalize_boolean(False)
    assert not normalize_boolean("false")
    with pytest.raises(ValueError, match="true/false"):
        normalize_boolean("not-a-boolean")
    with pytest.raises(ValueError, match="true/false"):
        normalize_boolean(1)


def test_aggregation_retains_failures_and_distinguishes_missing_runs(
    tmp_path: Path,
) -> None:
    _write_seed(
        tmp_path,
        92,
        [
            _candidate_row(92, "candidate_a", precision_at_10=0.6, best_epoch=2),
            _candidate_row(
                92,
                "candidate_b",
                top_fraction=0.02,
                valid="False",
                utility_floor_pass="False",
            ),
        ],
    )
    _write_seed(
        tmp_path,
        93,
        [_candidate_row(93, "candidate_a", precision_at_10=0.8, best_epoch=4)],
    )

    result = aggregate_all_candidates(tmp_path, expected_seeds=[92, 93])
    assert len(result.raw) == 3
    failed = result.raw.loc[result.raw[CONFIGURATION_NAME_COLUMN] == "candidate_b"].iloc[0]
    assert not bool(failed["valid"])
    assert failed["run_outcome"] == "invalid"
    assert failed["error_message"] == "synthetic failure"

    summary = result.summary.set_index(CONFIGURATION_NAME_COLUMN)
    candidate_a = summary.loc["candidate_a"]
    assert candidate_a["seed_count"] == 2
    assert candidate_a["invalid_count"] == 0
    assert candidate_a["missing_seed_count"] == 0
    assert candidate_a["precision_at_10_mean"] == pytest.approx(0.7)
    assert candidate_a["precision_at_10_std"] == pytest.approx(0.1)
    assert candidate_a["best_epoch_mean"] == pytest.approx(3.0)
    assert candidate_a["best_epoch_mode"] == 2
    assert candidate_a["teacher_batch_size"] == 16
    assert candidate_a["fisher_retain_sample_size"] == 32

    candidate_b = summary.loc["candidate_b"]
    assert candidate_b["run_count"] == 1
    assert candidate_b["invalid_count"] == 1
    assert candidate_b["valid_count"] == 0
    assert candidate_b["valid_rate"] == 0.0
    assert candidate_b["missing_seed_count"] == 1
    assert json.loads(candidate_b["missing_seeds"]) == [93]
    assert result.metadata["invalid_run_count"] == 1


def test_candidate_only_fallback_records_fingerprint_scope(tmp_path: Path) -> None:
    _write_seed(
        tmp_path,
        92,
        [_candidate_row(92, "candidate")],
        effective_config=False,
    )
    result = aggregate_all_candidates(tmp_path)
    assert set(result.raw["configuration_fingerprint_scope"]) == {"candidate_only"}
    assert result.metadata["fingerprint_scope_counts"] == {"candidate_only": 1}
    assert pd.isna(result.summary.iloc[0]["fisher_batch_size"])


def test_duplicate_seed_and_semantic_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    _write_seed(
        tmp_path,
        92,
        [
            _candidate_row(92, "alias_a"),
            _candidate_row(92, "alias_b"),
        ],
    )
    with pytest.raises(ValueError, match="duplicati per seed"):
        aggregate_all_candidates(tmp_path)


def test_same_name_with_different_effective_configuration_is_rejected(
    tmp_path: Path,
) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "same_name", top_fraction=0.01)])
    _write_seed(tmp_path, 93, [_candidate_row(93, "same_name", top_fraction=0.02)])
    with pytest.raises(ValueError, match="configuration_name"):
        aggregate_all_candidates(tmp_path)


def test_same_fingerprint_with_different_display_names_is_rejected(
    tmp_path: Path,
) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "alias_a")])
    _write_seed(tmp_path, 93, [_candidate_row(93, "alias_b")])
    with pytest.raises(ValueError, match="alias"):
        aggregate_all_candidates(tmp_path)


@pytest.mark.parametrize(
    ("mode", "status", "message"),
    [
        ("quick", "completed", "mode deve essere 'full'"),
        ("full", "running", "status deve essere 'completed'"),
    ],
)
def test_quick_or_incomplete_search_metadata_is_rejected(
    tmp_path: Path,
    mode: str,
    status: str,
    message: str,
) -> None:
    _write_seed(
        tmp_path,
        92,
        [_candidate_row(92, "candidate")],
        mode=mode,
        status=status,
    )
    with pytest.raises(ValueError, match=message):
        aggregate_all_candidates(tmp_path)


def test_expected_seeds_are_inferred_from_root_config(tmp_path: Path) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "candidate")])
    _write_seed(tmp_path, 93, [_candidate_row(93, "candidate")])
    _write_root_effective_config(tmp_path, [93, 92])
    assert infer_expected_seeds_from_root_config(tmp_path) == [92, 93]

    assert summarize_all_candidates.main(["--input-dir", str(tmp_path)]) == 0
    metadata = json.loads(
        (tmp_path / "all_candidates_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["expected_seeds"] == [92, 93]
    assert metadata["evidence_mode"] == "full"
    raw = pd.read_csv(tmp_path / "all_candidates_all_seeds.csv")
    assert set(raw["evidence_mode"]) == {"full"}


def test_cli_requires_exact_expected_seeds_when_root_config_is_missing(
    tmp_path: Path,
) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "candidate")])
    with pytest.raises(ValueError, match="--expected-seeds"):
        summarize_all_candidates.main(["--input-dir", str(tmp_path)])


def test_cli_count_only_mode_does_not_require_root_seed_config(
    tmp_path: Path,
) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "candidate")])
    assert summarize_all_candidates.main(
        [
            "--input-dir",
            str(tmp_path),
            "--expected-seed-count",
            "1",
        ]
    ) == 0
    metadata = json.loads(
        (tmp_path / "all_candidates_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["expected_seed_count"] == 1


def test_output_paths_refuse_canonical_sources_protected_dirs_and_collisions(
    tmp_path: Path,
) -> None:
    _write_seed(tmp_path, 92, [_candidate_row(92, "candidate")])
    sources = discover_search_result_files(tmp_path)
    repository = summarize_all_candidates.REPOSITORY_ROOT
    safe = {
        "raw_path": tmp_path / "raw.csv",
        "summary_path": tmp_path / "summary.csv",
        "pareto_path": tmp_path / "pareto.csv",
        "metadata_path": tmp_path / "metadata.json",
    }

    canonical = repository / "configs" / "final_config.json"
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="final_config.json"):
        validate_aggregation_output_paths(
            input_dir=tmp_path,
            sources=sources,
            repository_root=repository,
            **{**safe, "raw_path": canonical},
        )
    assert canonical.read_bytes() == before

    protected_sources = (
        sources[0].comparison_path,
        sources[0].effective_config_path,
        sources[0].search_metadata_path,
    )
    for protected_source in protected_sources:
        assert protected_source is not None
        with pytest.raises(ValueError, match="evidenza sorgente"):
            validate_aggregation_output_paths(
                input_dir=tmp_path,
                sources=sources,
                repository_root=repository,
                **{**safe, "summary_path": protected_source},
            )

    for protected_directory in (
        repository / "outputs" / "final_run",
        repository / "submission",
    ):
        with pytest.raises(ValueError, match="percorso protetto"):
            validate_aggregation_output_paths(
                input_dir=tmp_path,
                sources=sources,
                repository_root=repository,
                **{**safe, "pareto_path": protected_directory / "evidence.csv"},
            )

    with pytest.raises(ValueError, match="devono essere distinte"):
        validate_aggregation_output_paths(
            input_dir=tmp_path,
            sources=sources,
            repository_root=repository,
            **{**safe, "metadata_path": safe["raw_path"]},
        )


def test_fingerprint_is_deterministic_and_normalizes_inactive_fields() -> None:
    base = _candidate_row(92, "display_name")
    changed_nonsemantic = {
        **base,
        "name": "another_name",
        "experimental_rationale": "ignored",
        "output_path": "ignored/too",
        "gradient_ascent_learning_rate": 0.9,
        "gradient_ascent_batch_size": 999,
        "gradient_ascent_retain_distillation_weight": 99.0,
        "batchnorm_recalibration_batch_size": 999,
    }
    assert configuration_fingerprint(base) == configuration_fingerprint(
        changed_nonsemantic
    )

    active_ga = {**base, "gradient_ascent_steps": 2}
    changed_active_ga = {**active_ga, "gradient_ascent_learning_rate": 2e-5}
    assert configuration_fingerprint(active_ga) != configuration_fingerprint(
        changed_active_ga
    )
    with_shared = {**base, "teacher_batch_size": 16}
    changed_shared = {**base, "teacher_batch_size": 32}
    assert configuration_fingerprint(with_shared) != configuration_fingerprint(
        changed_shared
    )


def test_pareto_analysis_matches_known_frontier() -> None:
    summary = pd.DataFrame(
        [
            {
                CONFIGURATION_NAME_COLUMN: "a",
                CONFIGURATION_FINGERPRINT_COLUMN: "fa",
                "run_count": 2,
                "valid_count": 2,
                "utility_floor_pass_count": 2,
                "precision_at_10_mean": 0.9,
                "local_privacy_proxy_mean": 0.9,
                "execution_time_seconds_mean": 1.0,
                "selected_parameter_fraction_mean": 0.1,
            },
            {
                CONFIGURATION_NAME_COLUMN: "b",
                CONFIGURATION_FINGERPRINT_COLUMN: "fb",
                "run_count": 2,
                "valid_count": 2,
                "utility_floor_pass_count": 2,
                "precision_at_10_mean": 0.8,
                "local_privacy_proxy_mean": 0.8,
                "execution_time_seconds_mean": 2.0,
                "selected_parameter_fraction_mean": 0.2,
            },
            {
                CONFIGURATION_NAME_COLUMN: "c",
                CONFIGURATION_FINGERPRINT_COLUMN: "fc",
                "run_count": 2,
                "valid_count": 2,
                "utility_floor_pass_count": 2,
                "precision_at_10_mean": 0.95,
                "local_privacy_proxy_mean": 0.7,
                "execution_time_seconds_mean": 0.5,
                "selected_parameter_fraction_mean": 0.05,
            },
            {
                CONFIGURATION_NAME_COLUMN: "invalid",
                CONFIGURATION_FINGERPRINT_COLUMN: "fi",
                "run_count": 2,
                "valid_count": 1,
                "utility_floor_pass_count": 1,
                "precision_at_10_mean": 1.0,
                "local_privacy_proxy_mean": 1.0,
                "execution_time_seconds_mean": 0.1,
                "selected_parameter_fraction_mean": 0.01,
            },
        ]
    )
    pareto = pareto_analysis(summary).set_index(CONFIGURATION_NAME_COLUMN)
    assert set(pareto.index) == {"a", "b", "c"}
    assert bool(pareto.loc["a", "is_pareto_optimal"])
    assert bool(pareto.loc["c", "is_pareto_optimal"])
    assert not bool(pareto.loc["b", "is_pareto_optimal"])
    assert pareto.loc["b", "pareto_dominated_count"] == 1
