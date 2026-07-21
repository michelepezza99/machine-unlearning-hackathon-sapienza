"""Test degli invarianti di caricamento e split."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from machine_unlearning.data import (
    load_challenge_data,
    save_validation_ids,
    validate_data_model_compatibility,
)


def _copied_data_dir(
    synthetic_data_dir: Path, tmp_path: Path, name: str
) -> Path:
    destination = tmp_path / name
    shutil.copytree(synthetic_data_dir, destination)
    return destination


def _keep_one_positive_target(
    data_dir: Path, *, target_column: str, positive_user_id: int
) -> None:
    for shard in data_dir.glob("*c000.csv"):
        frame = pd.read_csv(shard, sep=";")
        frame[target_column] = (frame["user_id"] == positive_user_id).astype(int)
        frame.to_csv(shard, sep=";", index=False)
    forget_path = data_dir / "forget_data.csv"
    forget_frame = pd.read_csv(forget_path)
    forget_frame[target_column] = 0
    forget_frame.to_csv(forget_path, index=False)


def test_split_is_deterministic_and_disjoint(synthetic_data_dir: Path) -> None:
    first = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=19)
    second = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=19)
    assert first.validation_ids.tolist() == second.validation_ids.tolist()

    retain_ids = set(first.retain_train_frame["user_id"])
    validation_ids = set(first.validation_frame["user_id"])
    forget_ids = set(first.forget_frame["user_id"])
    assert not retain_ids & validation_ids
    assert not retain_ids & forget_ids
    assert not validation_ids & forget_ids


def test_validation_id_file_has_exact_schema(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=19)
    path = save_validation_ids(data, tmp_path / "validation_ids.csv")
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["user_id"]
    assert saved["user_id"].tolist() == data.validation_ids.tolist()


def test_duplicate_ids_are_rejected(synthetic_data_dir: Path, tmp_path: Path) -> None:
    duplicate_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "duplicate_data")
    shard = duplicate_dir / "synthetic_part-00000-c000.csv"
    frame = pd.read_csv(shard, sep=";")
    frame.loc[1, "user_id"] = frame.loc[0, "user_id"]
    frame.to_csv(shard, sep=";", index=False)
    with pytest.raises(ValueError, match="duplicati"):
        load_challenge_data(duplicate_dir, validation_fraction=0.2, seed=19)


def test_missing_user_ids_are_rejected(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "missing_user_id")
    shard = data_dir / "synthetic_part-00000-c000.csv"
    frame = pd.read_csv(shard, sep=";")
    frame.loc[0, "user_id"] = np.nan
    frame.to_csv(shard, sep=";", index=False)

    with pytest.raises(ValueError, match="mancanti"):
        load_challenge_data(data_dir, validation_fraction=0.2, seed=19)


def test_forget_ids_missing_from_training_are_rejected(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "missing_forget_id")
    forget_path = data_dir / "forget_data.csv"
    forget_frame = pd.read_csv(forget_path)
    forget_frame.loc[0, "user_id"] = 999_999
    forget_frame.to_csv(forget_path, index=False)

    with pytest.raises(ValueError, match="ID forget"):
        load_challenge_data(data_dir, validation_fraction=0.2, seed=19)


def test_nonfinite_features_are_replaced_and_column_order_is_preserved(
    synthetic_data_dir: Path,
) -> None:
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=19)

    assert sum(data.replaced_feature_values.values()) == 1
    assert np.isfinite(data.x_retain_train).all()
    assert np.isfinite(data.x_validation).all()
    assert np.isfinite(data.x_forget).all()
    assert data.schema.feature_columns == tuple(f"feature_{index}" for index in range(4))
    assert data.schema.target_columns == tuple(f"target__{index}" for index in range(12))


def test_invalid_targets_are_rejected(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "invalid_target")
    shard = data_dir / "synthetic_part-00000-c000.csv"
    frame = pd.read_csv(shard, sep=";")
    frame.loc[1, "target__0"] = 2
    frame.to_csv(shard, sep=";", index=False)

    with pytest.raises(ValueError, match="esclusivamente 0 e 1"):
        load_challenge_data(data_dir, validation_fraction=0.2, seed=19)


def test_missing_training_positives_report_target_names(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "rare_training_target")
    _keep_one_positive_target(
        data_dir, target_column="target__0", positive_user_id=1001
    )

    # With seed 2 the sole positive example is assigned to validation.
    with pytest.raises(ValueError, match=r"retain training.*target__0"):
        load_challenge_data(data_dir, validation_fraction=0.2, seed=2)


def test_missing_validation_positives_emit_named_warning(
    synthetic_data_dir: Path, tmp_path: Path
) -> None:
    data_dir = _copied_data_dir(synthetic_data_dir, tmp_path, "rare_validation_target")
    _keep_one_positive_target(
        data_dir, target_column="target__0", positive_user_id=1001
    )

    # With seed 11 the sole positive remains in retain training.
    with pytest.warns(RuntimeWarning, match=r"validation.*target__0"):
        data = load_challenge_data(data_dir, validation_fraction=0.2, seed=11)
    assert data.y_retain_train[:, 0].sum() == 1
    assert data.y_validation[:, 0].sum() == 0


def test_model_compatibility_checks_optional_column_order(
    synthetic_data_dir: Path,
) -> None:
    data = load_challenge_data(synthetic_data_dir, validation_fraction=0.2, seed=19)
    architecture = {"input_dim": 4, "num_outputs": 12}
    validate_data_model_compatibility(
        data,
        architecture,
        feature_columns=data.schema.feature_columns,
        target_columns=data.schema.target_columns,
    )

    with pytest.raises(ValueError, match="Ordine delle feature"):
        validate_data_model_compatibility(
            data,
            architecture,
            feature_columns=tuple(reversed(data.schema.feature_columns)),
            target_columns=data.schema.target_columns,
        )
    with pytest.raises(ValueError, match="Ordine delle target"):
        validate_data_model_compatibility(
            data,
            architecture,
            feature_columns=data.schema.feature_columns,
            target_columns=tuple(reversed(data.schema.target_columns)),
        )
