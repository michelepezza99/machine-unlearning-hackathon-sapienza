"""Smoke test end-to-end dello stesso workflow pubblico usato da main.py."""

from __future__ import annotations

import json
from pathlib import Path

from machine_unlearning.workflow import run_final_workflow


def test_synthetic_final_workflow(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    output_dir = tmp_path / "diagnostics"
    result = run_final_workflow(
        data_dir=synthetic_data_dir,
        output_dir=output_dir,
        submission_dir=tmp_path / "submission",
        config_path=synthetic_final_config,
        device_name="cpu",
    )
    assert result.method == "retraining_from_scratch"
    assert result.execution_time_seconds >= 0
    assert result.validation["inference_checked"] is True
    assert result.metrics["validation_precision_at_10"] >= 0
    metadata = json.loads(
        (output_dir / "method_metadata.json").read_text(encoding="utf-8")
    )
    assert "costruzione del modello" in metadata["timing_policy"]
    assert "optimizer" in metadata["timing_policy"]


def test_reused_output_removes_only_owned_stale_diagnostics(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    output_dir = tmp_path / "diagnostics"
    output_dir.mkdir()
    stale_history = output_dir / "final_gradient_ascent_history.csv"
    stale_history.write_text("stale", encoding="utf-8")
    unrelated_file = output_dir / "user_notes.txt"
    unrelated_file.write_text("preserve me", encoding="utf-8")

    run_final_workflow(
        data_dir=synthetic_data_dir,
        output_dir=output_dir,
        submission_dir=tmp_path / "submission",
        config_path=synthetic_final_config,
        device_name="cpu",
    )

    assert not stale_history.exists()
    assert (output_dir / "final_training_history.csv").is_file()
    assert unrelated_file.read_text(encoding="utf-8") == "preserve me"


def test_synthetic_hybrid_workflow(
    tmp_path: Path,
    synthetic_data_dir: Path,
) -> None:
    config_path = tmp_path / "hybrid_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "synthetic_hybrid",
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
                "gradient_ascent_steps": 1,
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
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "hybrid_diagnostics"
    result = run_final_workflow(
        data_dir=synthetic_data_dir,
        output_dir=output_dir,
        submission_dir=tmp_path / "hybrid_submission",
        config_path=config_path,
        device_name="cpu",
    )
    assert result.method == "hybrid_fisher_dampening"
    assert result.validation["inference_checked"] is True
    assert sorted(path.name for path in result.submission_dir.iterdir()) == [
        "execution_time.txt",
        "model_artifact",
        "validation_ids.csv",
    ]
    metadata = json.loads(
        (output_dir / "method_metadata.json").read_text(encoding="utf-8")
    )
    assert "ricostruzione del modello originale" in metadata["timing_policy"]
    assert "Fisher retain/forget" in metadata["timing_policy"]
