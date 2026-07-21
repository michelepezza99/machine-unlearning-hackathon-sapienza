"""Test della validazione anticipata della configurazione finale."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from machine_unlearning.training import MAX_RANDOM_SEED
from machine_unlearning.workflow import (
    load_json_config,
    run_final_workflow,
    validate_final_config,
)


def _retraining_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "retraining_from_scratch",
        "seed": 11,
        "validation_fraction": 0.2,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "training_batch_size": 8,
        "fixed_epochs": 1,
        "gradient_clip": None,
        "evaluation_batch_size": 16,
    }


def _hybrid_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "method": "hybrid_fisher_dampening",
        "seed": 11,
        "validation_fraction": 0.2,
        "evaluation_batch_size": 16,
        "teacher_batch_size": 8,
        "fisher_retain_sample_size": 4,
        "fisher_forget_sample_size": 4,
        "fisher_batch_size": 2,
        "include_bias": False,
        "include_batchnorm_affine": False,
        "top_fraction": 0.1,
        "forget_absolute_quantile": 0.25,
        "minimum_dampening_factor": 0.8,
        "dampening_strength": 1.0,
        "fisher_ratio_power": 1.0,
        "gradient_ascent_steps": 0,
        "gradient_ascent_learning_rate": 1e-6,
        "gradient_ascent_batch_size": 4,
        "gradient_ascent_retain_distillation_weight": 1.0,
        "repair_learning_rate": 1e-4,
        "repair_weight_decay": 1e-4,
        "repair_batch_size": 8,
        "fixed_repair_epochs": 1,
        "supervised_loss_weight": 1.0,
        "distillation_weight": 0.5,
        "parameter_regularization_weight": 1e-4,
        "selected_parameter_weight": 1.0,
        "gradient_clip": 1.0,
        "freeze_selected_during_repair": True,
        "recalibrate_batchnorm": True,
        "batchnorm_recalibration_batch_size": 8,
    }


def test_non_object_json_is_rejected_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(TypeError, match="oggetto"):
        load_json_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fixed_epochs", 0),
        ("training_batch_size", 0),
        ("learning_rate", 0.0),
        ("weight_decay", -0.1),
        ("validation_fraction", 1.0),
        ("evaluation_batch_size", 0),
        ("seed", -1),
        ("seed", MAX_RANDOM_SEED + 1),
    ],
)
def test_retraining_config_rejects_invalid_numeric_ranges(
    key: str, value: int | float
) -> None:
    config = _retraining_config()
    config[key] = value
    with pytest.raises((TypeError, ValueError), match=key):
        validate_final_config(config)


def test_invalid_config_fails_before_data_loading(tmp_path: Path) -> None:
    config = _retraining_config()
    config["fixed_epochs"] = 0
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output_dir = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="fixed_epochs"):
        run_final_workflow(
            data_dir=tmp_path / "missing_data",
            output_dir=output_dir,
            submission_dir=tmp_path / "submission",
            config_path=config_path,
            device_name="cpu",
        )
    assert not output_dir.exists()


@pytest.mark.parametrize("seed_override", [-1, MAX_RANDOM_SEED + 1])
def test_invalid_seed_override_fails_before_data_loading(
    tmp_path: Path, seed_override: int
) -> None:
    config_path = tmp_path / "valid.json"
    config_path.write_text(json.dumps(_retraining_config()), encoding="utf-8")
    output_dir = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="seed"):
        run_final_workflow(
            data_dir=tmp_path / "missing_data",
            output_dir=output_dir,
            submission_dir=tmp_path / "submission",
            config_path=config_path,
            seed_override=seed_override,
            device_name="cpu",
        )
    assert not output_dir.exists()


def test_diagnostic_cleanup_refuses_directories_without_partial_deletion(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "valid.json"
    config_path.write_text(json.dumps(_retraining_config()), encoding="utf-8")
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    stale_file = output_dir / "final_metrics.json"
    stale_file.write_text("stale", encoding="utf-8")
    blocking_directory = output_dir / "method_metadata.json"
    blocking_directory.mkdir()

    with pytest.raises(RuntimeError, match="non sono file"):
        run_final_workflow(
            data_dir=tmp_path / "missing_data",
            output_dir=output_dir,
            submission_dir=tmp_path / "submission",
            config_path=config_path,
            device_name="cpu",
        )
    assert stale_file.read_text(encoding="utf-8") == "stale"
    assert blocking_directory.is_dir()


def test_hybrid_config_requires_every_runtime_field() -> None:
    config = _hybrid_config()
    del config["teacher_batch_size"]
    with pytest.raises(KeyError, match="teacher_batch_size"):
        validate_final_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("fisher_retain_sample_size", 0),
        ("top_fraction", 0.0),
        ("forget_absolute_quantile", 1.0),
        ("minimum_dampening_factor", 1.1),
        ("gradient_ascent_steps", -1),
        ("gradient_clip", 0.0),
    ],
)
def test_hybrid_config_rejects_invalid_numeric_ranges(
    key: str, value: int | float
) -> None:
    config = _hybrid_config()
    config[key] = value
    with pytest.raises((TypeError, ValueError), match=key):
        validate_final_config(config)


def test_valid_configs_are_returned_without_mutation() -> None:
    retraining = _retraining_config()
    hybrid = _hybrid_config()
    assert validate_final_config(retraining) == retraining
    assert validate_final_config(hybrid) == hybrid
