"""Smoke test del workflow di ricerca separato."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_synthetic_search_proposes_fixed_config(
    tmp_path: Path,
    synthetic_data_dir: Path,
) -> None:
    search_config = {
        "schema_version": 1,
        "seed": 11,
        "validation_fraction": 0.2,
        "evaluation_batch_size": 16,
        "utility_floor_ratio": 0.5,
        "add_gradient_ascent_variants": 1,
        "retraining": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "training_batch_size": 8,
            "max_epochs": 3,
            "patience": 1,
        },
        "fisher": {
            "teacher_batch_size": 8,
            "fisher_retain_sample_size": 64,
            "fisher_forget_sample_size": 64,
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
            "repair_max_epochs": 3,
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
                "name": "synthetic_ssd",
                "top_fraction": 0.1,
                "minimum_dampening_factor": 0.8,
            }
        ],
    }
    search_config_path = tmp_path / "search.json"
    selected_config_path = tmp_path / "selected.json"
    search_config_path.write_text(json.dumps(search_config), encoding="utf-8")
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/search_configs.py",
            "--data-dir",
            str(synthetic_data_dir),
            "--config",
            str(search_config_path),
            "--output-dir",
            str(tmp_path / "search_outputs"),
            "--proposed-config",
            str(selected_config_path),
            "--quick",
            "--device",
            "cpu",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    selected = json.loads(selected_config_path.read_text(encoding="utf-8"))
    assert selected["method"] in {
        "retraining_from_scratch",
        "hybrid_fisher_dampening",
    }
    if selected["method"] == "retraining_from_scratch":
        assert selected["fixed_epochs"] >= 1
    search_output = tmp_path / "search_outputs"
    expected_files = {
        "effective_search_config.json",
        "search_comparison.csv",
        "finalists.csv",
        "search_metadata.json",
        "best_candidate_summary.json",
        "best_hybrid_repair_history.csv",
        "best_hybrid_gradient_ascent_history.csv",
    }
    assert expected_files <= {path.name for path in search_output.iterdir()}
    effective = json.loads(
        (search_output / "effective_search_config.json").read_text(encoding="utf-8")
    )
    assert effective["retraining"]["max_epochs"] == 2
    assert effective["retraining"]["patience"] == 1
    assert effective["fisher"]["fisher_retain_sample_size"] == 32
    assert effective["fisher"]["fisher_forget_sample_size"] == 32
    assert effective["common_candidate"]["repair_max_epochs"] == 1
    assert effective["common_candidate"]["repair_patience"] == 1
    assert effective["add_gradient_ascent_variants"] == 0
    metadata = json.loads(
        (search_output / "search_metadata.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (search_output / "best_candidate_summary.json").read_text(
            encoding="utf-8"
        )
    )
    for evidence in (metadata, summary):
        assert evidence["device"] == "cpu"
        assert set(evidence["original_model_metrics"]) == {
            "validation_precision_at_10",
            "validation_bce",
            "forget_bce",
        }
        assert evidence["utility_floor_precision_at_10"] >= 0.0
