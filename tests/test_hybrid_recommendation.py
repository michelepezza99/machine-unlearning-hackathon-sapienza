"""Test sintetici della raccomandazione ibrida multi-seed rigorosa."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from machine_unlearning.hybrid_recommendation import (
    RecommendationResult,
    recommend_hybrid_configuration,
    select_positive_epoch_mode,
)
from machine_unlearning.search import merge_candidate_configs
from machine_unlearning.search_aggregation import configuration_fingerprint
from machine_unlearning.workflow import validate_final_config
from scripts import select_final_hybrid


EXPECTED_SEEDS = (92, 93, 94)


def _search_config(
    *, candidates: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    config = {
        "schema_version": 1,
        "seed": 92,
        "validation_fraction": 0.11,
        "evaluation_batch_size": 64,
        "utility_floor_ratio": 0.985,
        "add_gradient_ascent_variants": 0,
        "retraining": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "training_batch_size": 16,
            "max_epochs": 3,
            "patience": 1,
        },
        "fisher": {
            "teacher_batch_size": 64,
            "fisher_retain_sample_size": 32,
            "fisher_forget_sample_size": 32,
            "fisher_batch_size": 8,
            "include_bias": False,
            "include_batchnorm_affine": False,
        },
        "common_candidate": {
            "forget_absolute_quantile": 0.5,
            "dampening_strength": 1.0,
            "fisher_ratio_power": 1.0,
            "repair_learning_rate": 1e-4,
            "repair_weight_decay": 0.0,
            "repair_batch_size": 16,
            "repair_max_epochs": 3,
            "repair_patience": 1,
            "supervised_loss_weight": 1.0,
            "distillation_weight": 0.5,
            "parameter_regularization_weight": 1e-4,
            "selected_parameter_weight": 1.0,
            "gradient_clip": 1.0,
            "freeze_selected_during_repair": True,
            "gradient_ascent_steps": 0,
            "gradient_ascent_learning_rate": 1e-5,
            "gradient_ascent_batch_size": 8,
            "gradient_ascent_retain_distillation_weight": 1.0,
            "recalibrate_batchnorm": False,
            "batchnorm_recalibration_batch_size": 64,
        },
        "candidates": [
            {
                "name": "candidate",
                "top_fraction": 0.01,
                "minimum_dampening_factor": 0.9,
            }
        ],
    }
    if candidates is not None:
        config["candidates"] = candidates
    return config


def _candidate_fingerprint(config: dict[str, Any], name: str) -> str:
    candidate = next(
        candidate
        for candidate in merge_candidate_configs(config)
        if candidate["name"] == name
    )
    return configuration_fingerprint(
        {
            "validation_fraction": config["validation_fraction"],
            "evaluation_batch_size": config["evaluation_batch_size"],
            "utility_floor_ratio": config["utility_floor_ratio"],
            **config["fisher"],
            **candidate,
        }
    )


def _runtime_fields(*, top_fraction: float = 0.01) -> dict[str, Any]:
    return {
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
        "repair_batch_size": 16,
        "supervised_loss_weight": 1.0,
        "distillation_weight": 0.5,
        "parameter_regularization_weight": 1e-4,
        "selected_parameter_weight": 1.0,
        "gradient_clip": 1.0,
        "freeze_selected_during_repair": True,
        "recalibrate_batchnorm": False,
        "batchnorm_recalibration_batch_size": 64,
    }


def _raw_rows(
    name: str,
    fingerprint: str,
    *,
    seeds: tuple[int, ...] = EXPECTED_SEEDS,
    epochs: tuple[int, ...] = (2, 3, 2),
    floor_fail_seed: int | None = None,
    top_fraction: float = 0.01,
    privacy: float = 0.7,
    precision: float = 0.04,
    utility: float = 0.99,
    execution_time: float = 10.0,
    selected_fraction: float = 0.01,
    fingerprint_scope: str = "candidate_and_shared",
    evidence_mode: str = "full",
) -> list[dict[str, Any]]:
    assert len(seeds) == len(epochs)
    return [
        {
            "seed": seed,
            "configuration_name": name,
            "configuration_fingerprint": fingerprint,
            "configuration_fingerprint_scope": fingerprint_scope,
            "evidence_mode": evidence_mode,
            "valid": True,
            "utility_floor_pass": seed != floor_fail_seed,
            "best_epoch": epoch,
            "local_privacy_proxy": privacy,
            "precision_at_10": precision,
            "utility_ratio": utility,
            "execution_time_seconds": execution_time,
            "selected_parameter_fraction": selected_fraction,
            **_runtime_fields(top_fraction=top_fraction),
        }
        for seed, epoch in zip(seeds, epochs, strict=True)
    ]


def _summary_row(
    name: str,
    fingerprint: str,
    *,
    seed_count: int = 3,
    valid_rate: float = 1.0,
    floor_rate: float = 1.0,
    privacy_min: float = 0.7,
    privacy_mean: float = 0.7,
    precision_min: float = 0.04,
    utility_min: float = 0.99,
    time_mean: float = 10.0,
    time_max: float = 10.0,
    selected_fraction: float = 0.01,
    best_epoch_mode: int = 2,
) -> dict[str, Any]:
    return {
        "configuration_name": name,
        "configuration_fingerprint": fingerprint,
        "seed_count": seed_count,
        "valid_rate": valid_rate,
        "utility_floor_pass_rate": floor_rate,
        "local_privacy_proxy_min": privacy_min,
        "local_privacy_proxy_mean": privacy_mean,
        "precision_at_10_min": precision_min,
        "utility_ratio_min": utility_min,
        "execution_time_seconds_mean": time_mean,
        "execution_time_seconds_max": time_max,
        "selected_parameter_fraction_mean": selected_fraction,
        "best_epoch_mode": best_epoch_mode,
    }


def _recommend(
    summary_rows: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    search_config: dict[str, Any] | None = None,
) -> RecommendationResult:
    return recommend_hybrid_configuration(
        pd.DataFrame(summary_rows),
        pd.DataFrame(raw_rows),
        search_config or _search_config(),
        expected_seeds=EXPECTED_SEEDS,
        final_seed=92,
    )


def test_exact_expected_seed_set_rejects_same_count_with_wrong_seed() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint)],
        _raw_rows(
            "candidate",
            fingerprint,
            seeds=(92, 93, 95),
            epochs=(2, 2, 2),
        ),
    )

    assert result.final_config is None
    assert result.recommendation["status"] == "provisional_no_safe_selection"
    diagnostic = result.diagnostics.iloc[0]
    assert bool(diagnostic["expected_seed_count_pass"]) is True
    assert bool(diagnostic["expected_seed_set_pass"]) is False


def test_missing_seed_is_not_eligible() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint, seed_count=2)],
        _raw_rows("candidate", fingerprint, seeds=(92, 93), epochs=(2, 2)),
    )

    assert result.final_config is None
    diagnostic = result.diagnostics.iloc[0]
    assert bool(diagnostic["expected_seed_count_pass"]) is False
    assert bool(diagnostic["expected_seed_set_pass"]) is False


def test_one_utility_floor_failure_is_not_silently_relaxed() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint, floor_rate=2 / 3)],
        _raw_rows("candidate", fingerprint, floor_fail_seed=94),
    )

    assert result.final_config is None
    assert result.recommendation["eligible_candidate_count"] == 0
    assert bool(result.diagnostics.iloc[0]["utility_floor_pass_rate_pass"]) is False


def test_hierarchical_ranking_is_deterministic() -> None:
    config = _search_config(
        candidates=[
            {
                "name": "fast_but_weaker_privacy",
                "top_fraction": 0.005,
                "minimum_dampening_factor": 0.9,
            },
            {
                "name": "slower_stronger_privacy",
                "top_fraction": 0.02,
                "minimum_dampening_factor": 0.9,
            },
        ]
    )
    fast_fingerprint = _candidate_fingerprint(
        config, "fast_but_weaker_privacy"
    )
    private_fingerprint = _candidate_fingerprint(
        config, "slower_stronger_privacy"
    )
    summaries = [
        _summary_row(
            "fast_but_weaker_privacy",
            fast_fingerprint,
            privacy_min=0.6,
            privacy_mean=0.6,
            time_mean=1.0,
            time_max=1.0,
        ),
        _summary_row(
            "slower_stronger_privacy",
            private_fingerprint,
            privacy_min=0.7,
            privacy_mean=0.7,
            time_mean=20.0,
            time_max=20.0,
        ),
    ]
    raw = [
        *_raw_rows(
            "fast_but_weaker_privacy",
            fast_fingerprint,
            top_fraction=0.005,
            privacy=0.6,
            execution_time=1.0,
        ),
        *_raw_rows(
            "slower_stronger_privacy",
            private_fingerprint,
            top_fraction=0.02,
            privacy=0.7,
            execution_time=20.0,
        ),
    ]

    forward = _recommend(summaries, raw, search_config=config)
    reverse = _recommend(
        list(reversed(summaries)), list(reversed(raw)), search_config=config
    )

    assert forward.recommendation["selected"]["configuration_name"] == (
        "slower_stronger_privacy"
    )
    assert reverse.recommendation["selected"] == forward.recommendation["selected"]
    assert forward.diagnostics.iloc[0]["configuration_name"] == (
        "slower_stronger_privacy"
    )
    assert forward.diagnostics["diagnostic_rank"].tolist() == [1, 2]


def test_positive_epoch_mode_uses_smaller_positive_tie_and_rejects_zero_only() -> None:
    selected, observed, counts = select_positive_epoch_mode([3, 2, 3, 2, 4])
    assert selected == 2
    assert observed == [3, 2, 3, 2, 4]
    assert counts == {2: 2, 3: 2, 4: 1}
    assert select_positive_epoch_mode([0, 4, 4, 2])[0] == 4
    assert select_positive_epoch_mode([0, 0, 2, 2])[0] == 2

    with pytest.raises(ValueError, match="moda positiva"):
        select_positive_epoch_mode([0, 0, 2])


def test_unique_zero_epoch_mode_is_provisional() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint, best_epoch_mode=0)],
        _raw_rows("candidate", fingerprint, epochs=(0, 0, 2)),
    )

    assert result.final_config is None
    assert result.recommendation["final_config_generated"] is False
    assert bool(result.diagnostics.iloc[0]["positive_epoch_available"]) is False


def test_generated_hybrid_config_passes_authoritative_validator() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint)],
        _raw_rows("candidate", fingerprint, epochs=(2, 3, 2)),
    )

    assert result.final_config is not None
    assert validate_final_config(result.final_config) == result.final_config
    assert result.final_config["method"] == "hybrid_fisher_dampening"
    assert result.final_config["fixed_repair_epochs"] == 2
    assert result.final_config["seed"] == 92
    assert result.final_config["selection_status"].startswith("provisional")
    assert result.recommendation["final_config_generated"] is True
    assert "final_config_written" not in result.recommendation


def test_search_config_fingerprint_mismatch_is_not_eligible() -> None:
    evidence_config = _search_config()
    fingerprint = _candidate_fingerprint(evidence_config, "candidate")
    mismatched_config = deepcopy(evidence_config)
    mismatched_config["fisher"]["fisher_retain_sample_size"] = 64

    result = _recommend(
        [_summary_row("candidate", fingerprint)],
        _raw_rows("candidate", fingerprint),
        search_config=mismatched_config,
    )

    assert result.final_config is None
    assert bool(result.diagnostics.iloc[0]["search_config_membership_pass"]) is False


@pytest.mark.parametrize(
    ("raw_override", "diagnostic_column"),
    [
        ({"fingerprint_scope": "candidate_only"}, "full_fingerprint_scope_pass"),
        ({"evidence_mode": "quick"}, "full_evidence_mode_pass"),
    ],
)
def test_non_full_or_candidate_only_evidence_is_not_eligible(
    raw_override: dict[str, str], diagnostic_column: str
) -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    result = _recommend(
        [_summary_row("candidate", fingerprint)],
        _raw_rows("candidate", fingerprint, **raw_override),
    )

    assert result.final_config is None
    assert bool(result.diagnostics.iloc[0][diagnostic_column]) is False


def test_stale_tampered_summary_cannot_select_candidate() -> None:
    config = _search_config(
        candidates=[
            {
                "name": "tampered",
                "top_fraction": 0.01,
                "minimum_dampening_factor": 0.9,
            },
            {
                "name": "honest",
                "top_fraction": 0.02,
                "minimum_dampening_factor": 0.9,
            },
        ]
    )
    tampered_fingerprint = _candidate_fingerprint(config, "tampered")
    honest_fingerprint = _candidate_fingerprint(config, "honest")
    summaries = [
        _summary_row(
            "tampered",
            tampered_fingerprint,
            privacy_min=0.99,
            privacy_mean=0.99,
        ),
        _summary_row(
            "honest",
            honest_fingerprint,
            privacy_min=0.8,
            privacy_mean=0.8,
        ),
    ]
    raw = [
        *_raw_rows("tampered", tampered_fingerprint, privacy=0.5),
        *_raw_rows(
            "honest",
            honest_fingerprint,
            privacy=0.8,
            top_fraction=0.02,
        ),
    ]

    result = _recommend(summaries, raw, search_config=config)

    assert result.recommendation["selected"]["configuration_name"] == "honest"
    tampered_diagnostic = result.diagnostics.loc[
        result.diagnostics["configuration_name"] == "tampered"
    ].iloc[0]
    assert bool(tampered_diagnostic["summary_raw_evidence_match"]) is False
    assert bool(tampered_diagnostic["ranking_aggregates_match"]) is False


@pytest.mark.parametrize(
    "output_option",
    ["--config-output", "--recommendation-output", "--diagnostics-output"],
)
def test_cli_refuses_every_canonical_output_path_without_changing_bytes(
    output_option: str,
) -> None:
    canonical = select_final_hybrid.REPOSITORY_ROOT / "configs" / "final_config.json"
    before = canonical.read_bytes()

    with pytest.raises(ValueError, match="final_config.json"):
        select_final_hybrid.main([output_option, str(canonical)])

    assert canonical.read_bytes() == before


def test_cli_refuses_output_collision_and_input_overwrite(tmp_path: Path) -> None:
    collision = tmp_path / "same.json"
    with pytest.raises(ValueError, match="Collisione"):
        select_final_hybrid.main(
            [
                "--config-output",
                str(collision),
                "--recommendation-output",
                str(collision),
            ]
        )

    source = tmp_path / "source.csv"
    with pytest.raises(ValueError, match="sovrascrivere un input"):
        select_final_hybrid.main(
            ["--summary", str(source), "--config-output", str(source)]
        )


@pytest.mark.parametrize(
    "protected_path",
    [
        select_final_hybrid.REPOSITORY_ROOT
        / "outputs"
        / "final_run"
        / "recommendation.json",
        select_final_hybrid.REPOSITORY_ROOT / "submission" / "diagnostics.csv",
    ],
)
def test_cli_refuses_protected_final_artifact_directories(
    protected_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outputs/final_run o submission"):
        select_final_hybrid.main(["--config-output", str(protected_path)])


def _write_cli_evidence(
    tmp_path: Path, *, epochs: tuple[int, ...]
) -> tuple[Path, Path, Path]:
    config = _search_config()
    fingerprint = _candidate_fingerprint(config, "candidate")
    best_epoch_mode = min(
        epoch
        for epoch, count in Counter(epochs).items()
        if count == max(Counter(epochs).values())
    )
    summary_path = tmp_path / "summary.csv"
    raw_path = tmp_path / "raw.csv"
    config_path = tmp_path / "search.json"
    pd.DataFrame(
        [_summary_row("candidate", fingerprint, best_epoch_mode=best_epoch_mode)]
    ).to_csv(summary_path, index=False)
    pd.DataFrame(
        _raw_rows("candidate", fingerprint, epochs=epochs)
    ).to_csv(raw_path, index=False)
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return summary_path, raw_path, config_path


def test_cli_warns_and_preserves_stale_config_on_ineligible_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path, raw_path, search_path = _write_cli_evidence(
        tmp_path, epochs=(0, 0, 2)
    )
    config_output = tmp_path / "existing_config.json"
    stale_bytes = b'{"stale": true}\n'
    config_output.write_bytes(stale_bytes)
    recommendation_output = tmp_path / "recommendation.json"
    diagnostics_output = tmp_path / "diagnostics.csv"

    return_code = select_final_hybrid.main(
        [
            "--summary",
            str(summary_path),
            "--raw",
            str(raw_path),
            "--search-config",
            str(search_path),
            "--expected-seeds",
            "92",
            "93",
            "94",
            "--seed",
            "92",
            "--config-output",
            str(config_output),
            "--recommendation-output",
            str(recommendation_output),
            "--diagnostics-output",
            str(diagnostics_output),
        ]
    )

    assert return_code == 2
    assert config_output.read_bytes() == stale_bytes
    audit = json.loads(recommendation_output.read_text(encoding="utf-8"))
    assert audit["final_config_generated"] is False
    assert audit["final_config_written"] is False
    assert audit["stale_config_output_exists"] is True
    assert "obsoleto" in capsys.readouterr().err


def test_input_frames_are_not_mutated() -> None:
    fingerprint = _candidate_fingerprint(_search_config(), "candidate")
    summary = pd.DataFrame([_summary_row("candidate", fingerprint)])
    raw = pd.DataFrame(_raw_rows("candidate", fingerprint))
    expected_summary = deepcopy(summary)
    expected_raw = deepcopy(raw)

    recommend_hybrid_configuration(
        summary,
        raw,
        _search_config(),
        expected_seeds=EXPECTED_SEEDS,
        final_seed=92,
    )

    pd.testing.assert_frame_equal(summary, expected_summary)
    pd.testing.assert_frame_equal(raw, expected_raw)
