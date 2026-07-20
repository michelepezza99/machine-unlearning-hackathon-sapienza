"""Test della struttura e del validator della submission."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from machine_unlearning.submission import validate_submission
from machine_unlearning.workflow import run_final_workflow


def _build_submission(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> Path:
    submission_dir = tmp_path / "submission"
    run_final_workflow(
        data_dir=synthetic_data_dir,
        output_dir=tmp_path / "outputs",
        submission_dir=submission_dir,
        config_path=synthetic_final_config,
        device_name="cpu",
    )
    return submission_dir


def test_submission_has_exact_files_and_valid_content(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    result = validate_submission(submission_dir, data_dir=synthetic_data_dir)
    assert set(result["files"]) == {
        "model_artifact",
        "execution_time.txt",
        "validation_ids.csv",
    }
    assert (submission_dir / "execution_time.txt").read_text().isdigit()


def test_duplicate_validation_ids_are_rejected(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    validation_path = submission_dir / "validation_ids.csv"
    frame = pd.read_csv(validation_path)
    frame.loc[1, "user_id"] = frame.loc[0, "user_id"]
    frame.to_csv(validation_path, index=False)
    with pytest.raises(ValueError, match="duplicati"):
        validate_submission(submission_dir)


def test_unexpected_submission_file_is_rejected(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    (submission_dir / "notes.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        validate_submission(submission_dir)
