"""Fixture sintetiche condivise dai test."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from machine_unlearning.model import DynamicMLP, dynamic_mlp_source, model_state_to_cpu


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Crea due shard, un forget set e un artifact compatibile."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_count = 36
    output_count = 12
    generator = np.random.default_rng(123)
    frame = pd.DataFrame(
        {
            "feature_0": generator.normal(size=sample_count),
            "feature_1": generator.normal(size=sample_count),
            "feature_2": generator.normal(size=sample_count),
            "feature_3": generator.normal(size=sample_count),
        }
    )
    frame.loc[3, "feature_1"] = np.nan
    for target_index in range(output_count):
        frame[f"target__{target_index}"] = (
            (np.arange(sample_count) + target_index) % 3 != 0
        ).astype(int)
    frame["user_id"] = np.arange(1000, 1000 + sample_count)
    frame.iloc[:18].to_csv(
        data_dir / "synthetic_part-00000-c000.csv", sep=";", index=False
    )
    frame.iloc[18:].to_csv(
        data_dir / "synthetic_part-00001-c000.csv", sep=";", index=False
    )
    frame.iloc[[0, 5, 20, 30]].to_csv(data_dir / "forget_data.csv", index=False)

    torch.manual_seed(7)
    model = DynamicMLP(input_dim=4, hidden_layers=[6], num_outputs=output_count)
    payload = {
        "state_dict": model_state_to_cpu(model),
        "architecture": {
            "input_dim": 4,
            "hidden_layers": [6],
            "num_outputs": output_count,
        },
        "best_hyperparameters": {
            "lr": 0.001,
            "epochs": 2,
            "batch_size": 8,
        },
        "model_class_source": dynamic_mlp_source(),
    }
    with (data_dir / "model_artifact").open("wb") as handle:
        pickle.dump(payload, handle)
    return data_dir


@pytest.fixture
def synthetic_final_config(tmp_path: Path) -> Path:
    """Crea una configurazione finale veloce per lo smoke test."""
    path = tmp_path / "final_config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "synthetic_retraining",
                "method": "retraining_from_scratch",
                "seed": 11,
                "validation_fraction": 0.2,
                "optimizer": "adam",
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "training_batch_size": 8,
                "fixed_epochs": 1,
                "gradient_clip": 1.0,
                "evaluation_batch_size": 16,
            }
        ),
        encoding="utf-8",
    )
    return path
