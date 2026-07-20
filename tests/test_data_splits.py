"""Test degli invarianti di caricamento e split."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from machine_unlearning.data import load_challenge_data, save_validation_ids


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
    duplicate_dir = tmp_path / "duplicate_data"
    shutil.copytree(synthetic_data_dir, duplicate_dir)
    shard = duplicate_dir / "synthetic_part-00000-c000.csv"
    frame = pd.read_csv(shard, sep=";")
    frame.loc[1, "user_id"] = frame.loc[0, "user_id"]
    frame.to_csv(shard, sep=";", index=False)
    with pytest.raises(ValueError, match="duplicati"):
        load_challenge_data(duplicate_dir, validation_fraction=0.2, seed=19)
