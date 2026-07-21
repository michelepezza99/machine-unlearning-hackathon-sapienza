"""Test della struttura e del validator della submission."""

from __future__ import annotations

import pickle
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


def test_missing_submission_file_is_rejected(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    (submission_dir / "execution_time.txt").unlink()
    with pytest.raises(ValueError, match="mancanti"):
        validate_submission(submission_dir)


@pytest.mark.parametrize("invalid_time", ["-1", "1.5", "1\n2", "not-a-time"])
def test_invalid_execution_time_is_rejected(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
    invalid_time: str,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    (submission_dir / "execution_time.txt").write_text(
        invalid_time, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="intero non negativo"):
        validate_submission(submission_dir)


def test_validation_ids_must_match_reconstructed_split(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    validation_path = submission_dir / "validation_ids.csv"
    frame = pd.read_csv(validation_path)
    frame.loc[0, "user_id"] = int(frame["user_id"].max()) + 1_000_000
    frame.to_csv(validation_path, index=False)
    with pytest.raises(ValueError, match="split deterministico"):
        validate_submission(submission_dir, data_dir=synthetic_data_dir)


def test_artifact_column_order_must_match_original_data(
    tmp_path: Path,
    synthetic_data_dir: Path,
    synthetic_final_config: Path,
) -> None:
    submission_dir = _build_submission(
        tmp_path, synthetic_data_dir, synthetic_final_config
    )
    artifact_path = submission_dir / "model_artifact"
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    payload["feature_columns"] = list(reversed(payload["feature_columns"]))
    with artifact_path.open("wb") as handle:
        pickle.dump(payload, handle)

    with pytest.raises(ValueError, match="Ordine delle feature"):
        validate_submission(submission_dir, data_dir=synthetic_data_dir)
