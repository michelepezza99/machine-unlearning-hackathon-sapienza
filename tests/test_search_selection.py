"""Test rapidi per configurazione, ranking e reporting della ricerca."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from machine_unlearning.search import (
    MULTI_SEED_METRICS,
    build_effective_search_config,
    merge_candidate_configs,
    summarize_multi_seed_results,
    validate_search_config,
)
from machine_unlearning.unlearning import (
    progressive_search,
    select_best_search_result,
)
from scripts import search_configs


def _search_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "seed": 11,
        "validation_fraction": 0.2,
        "evaluation_batch_size": 16,
        "utility_floor_ratio": 0.9,
        "add_gradient_ascent_variants": 2,
        "quick": {
            "retraining_max_epochs": 2,
            "retraining_patience": 1,
            "fisher_retain_sample_size": 8,
            "fisher_forget_sample_size": 7,
            "candidate_count": 1,
            "repair_max_epochs": 1,
            "repair_patience": 1,
        },
        "retraining": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "training_batch_size": 8,
            "max_epochs": 10,
            "patience": 4,
        },
        "fisher": {
            "teacher_batch_size": 8,
            "fisher_retain_sample_size": 100,
            "fisher_forget_sample_size": 100,
            "fisher_batch_size": 2,
            "include_bias": False,
            "include_batchnorm_affine": False,
        },
        "common_candidate": {
            "forget_absolute_quantile": 0.25,
            "dampening_strength": 1.0,
            "fisher_ratio_power": 1.0,
            "repair_learning_rate": 0.0001,
            "repair_weight_decay": 0.0,
            "repair_batch_size": 8,
            "repair_max_epochs": 6,
            "repair_patience": 2,
            "supervised_loss_weight": 1.0,
            "distillation_weight": 0.5,
            "parameter_regularization_weight": 0.0001,
            "selected_parameter_weight": 1.0,
            "gradient_clip": 1.0,
            "freeze_selected_during_repair": True,
            "gradient_ascent_steps": 0,
            "gradient_ascent_learning_rate": 0.000001,
            "gradient_ascent_batch_size": 4,
            "gradient_ascent_retain_distillation_weight": 1.0,
            "recalibrate_batchnorm": False,
            "batchnorm_recalibration_batch_size": 8,
        },
        "candidates": [
            {
                "name": "candidate_a",
                "top_fraction": 0.1,
                "minimum_dampening_factor": 0.8,
                "repair_max_epochs": 99,
                "repair_patience": 99,
            },
            {
                "name": "candidate_b",
                "top_fraction": 0.2,
                "minimum_dampening_factor": 0.9,
            },
        ],
    }


def _result(
    name: str,
    *,
    score: float,
    privacy: float = 0.6,
    precision: float = 0.8,
    elapsed: float = 2.0,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "config": {"name": name},
        "config_index": 1,
        "status": "success" if valid else "failed",
        "valid": valid,
        "error_message": None if valid else "synthetic failure",
        "metrics": {
            "precision_at_10": precision,
            "validation_bce": 0.4,
            "forget_bce": 0.7,
            "local_privacy_proxy": privacy,
            "execution_time_seconds": elapsed,
            "local_search_score": score,
            "best_epoch": 1,
            "utility_floor_pass": True,
        },
    }


def test_quick_overrides_all_expensive_candidate_settings() -> None:
    effective = build_effective_search_config(
        _search_config(), quick=True, max_candidates=None
    )
    assert effective["retraining"]["max_epochs"] == 2
    assert effective["retraining"]["patience"] == 1
    assert effective["fisher"]["fisher_retain_sample_size"] == 8
    assert effective["fisher"]["fisher_forget_sample_size"] == 7
    assert effective["add_gradient_ascent_variants"] == 0
    assert len(effective["candidates"]) == 1
    merged = merge_candidate_configs(effective)
    assert merged[0]["repair_max_epochs"] == 1
    assert merged[0]["repair_patience"] == 1
    assert merged[0]["utility_floor_ratio"] == 0.9


def test_search_config_rejects_boolean_schema_and_non_mapping_candidate() -> None:
    boolean_schema = _search_config()
    boolean_schema["schema_version"] = True
    with pytest.raises(ValueError, match="schema_version"):
        validate_search_config(boolean_schema)
    invalid_candidate = _search_config()
    invalid_candidate["candidates"] = ["not an object"]
    with pytest.raises(TypeError, match=r"candidates\[0\]"):
        validate_search_config(invalid_candidate)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_search_config_rejects_seed_outside_shared_range(seed: int) -> None:
    config = _search_config()
    config["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        validate_search_config(config)


def test_selection_is_finite_aware_and_uses_every_tie_breaker() -> None:
    invalid = _result("invalid", score=float("nan"), valid=True)
    slower = _result("slower", score=0.8, privacy=0.7, precision=0.8, elapsed=4.0)
    faster = _result("faster", score=0.8, privacy=0.7, precision=0.8, elapsed=1.0)
    winner = select_best_search_result(
        [invalid, slower, faster],
        baseline_precision_at_10=0.8,
        utility_floor_ratio=0.9,
    )
    assert winner["config"]["name"] == "faster"


def test_progressive_search_records_failure_and_continues(monkeypatch) -> None:
    def fake_execute(configuration, **_kwargs):
        if configuration["name"] == "bad":
            raise RuntimeError("mask is empty")
        result = _result(configuration["name"], score=0.75)
        result.update(
            {
                "config": deepcopy(configuration),
                "mask_metadata": {
                    "selected": 2,
                    "selected_fraction_of_eligible": 0.25,
                },
                "dampening_metadata": {},
                "repair_history": pd.DataFrame([{"epoch": 0}]),
                "gradient_ascent_history": pd.DataFrame(),
            }
        )
        return result

    monkeypatch.setattr(
        "machine_unlearning.unlearning.execute_search_candidate", fake_execute
    )
    best, comparison, results = progressive_search(
        [
            {"name": "bad", "repair_learning_rate": 0.001},
            {"name": "good", "repair_learning_rate": 0.001},
        ],
        execute_kwargs={"seed": 11},
        baseline_precision_at_10=0.8,
        utility_floor_ratio=0.9,
        add_gradient_ascent_variants=0,
    )
    assert best is not None and best["config"]["name"] == "good"
    assert len(results) == 2
    failure = comparison.loc[comparison["name"] == "bad"].iloc[0]
    assert failure["status"] == "failed"
    assert failure["error_type"] == "RuntimeError"
    assert "mask is empty" in failure["error_message"]


def test_multi_seed_summary_has_mean_std_min_and_max() -> None:
    rows = []
    for seed, offset in [(11, 0.0), (12, 0.2)]:
        row = {
            "seed": seed,
            "method": "retraining_from_scratch",
            "configuration": "retraining_reference",
            "valid": True,
        }
        row.update({metric: 1.0 + offset for metric in MULTI_SEED_METRICS})
        rows.append(row)
    summary = summarize_multi_seed_results(pd.DataFrame(rows)).iloc[0]
    for metric in MULTI_SEED_METRICS:
        assert summary[f"{metric}_mean"] == pytest.approx(1.1)
        assert summary[f"{metric}_std"] == pytest.approx(0.1)
        assert summary[f"{metric}_min"] == pytest.approx(1.0)
        assert summary[f"{metric}_max"] == pytest.approx(1.2)


def test_cli_quick_paths_and_canonical_safeguard() -> None:
    arguments = search_configs.build_parser().parse_args(["--quick"])
    output_dir, proposed = search_configs._resolve_output_paths(
        arguments, ["--quick"]
    )
    assert output_dir.as_posix() == "outputs/search/quick"
    assert proposed.as_posix() == "outputs/search/quick/proposed_final_config.json"
    with pytest.raises(ValueError, match="rifiutata"):
        search_configs._assert_safe_proposed_path(
            search_configs.CANONICAL_FINAL_CONFIG, allow_canonical=False
        )
    search_configs._assert_safe_proposed_path(
        search_configs.CANONICAL_FINAL_CONFIG, allow_canonical=True
    )


def test_zero_epoch_retraining_proposal_is_rejected() -> None:
    with pytest.raises(ValueError, match="almeno un'epoca"):
        search_configs._build_retraining_final_config(
            _search_config(), seed=11, best_epoch=0
        )
