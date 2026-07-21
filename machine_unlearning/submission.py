"""Creazione e validazione rigorosa dei tre file di submission."""

from __future__ import annotations

import math
import pickle
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .data import ID_COLUMN, load_challenge_data, validate_data_model_compatibility
from .metrics import predict_logits
from .model import (
    assert_finite_state,
    clone_state_dict_to_cpu,
    validate_artifact_payload,
)


SUBMISSION_FILENAMES = {
    "model_artifact",
    "execution_time.txt",
    "validation_ids.csv",
}


def create_submission(
    *,
    submission_dir: str | Path,
    final_state_dict: Mapping[str, torch.Tensor],
    execution_time_seconds: float,
    validation_ids: Sequence[Any] | np.ndarray,
    original_payload: Mapping[str, Any],
    final_config: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
    method_metadata: Mapping[str, Any],
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    seed: int,
    validation_fraction: float,
) -> dict[str, Path]:
    """Scrive i tre file finali senza lasciare contenuti estranei.

    Sovrascriviamo soltanto i tre nomi della challenge. Se la directory contiene
    altri file, interrompiamo l'esecuzione per non cancellare dati non riconosciuti.
    """
    if not math.isfinite(execution_time_seconds) or execution_time_seconds < 0:
        raise ValueError("Il tempo di esecuzione deve essere finito e non negativo.")
    validation_array = np.asarray(validation_ids)
    if validation_array.ndim != 1 or len(validation_array) == 0:
        raise ValueError("Gli ID di validation devono essere un vettore non vuoto.")
    validation_frame = pd.DataFrame({ID_COLUMN: validation_array})
    if validation_frame[ID_COLUMN].isna().any():
        raise ValueError("Gli ID di validation non possono essere mancanti.")
    if validation_frame[ID_COLUMN].duplicated().any():
        raise ValueError("Gli ID di validation non possono contenere duplicati.")

    destination = Path(submission_dir)
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = {entry.name for entry in destination.iterdir()} - SUBMISSION_FILENAMES
    if unexpected:
        raise RuntimeError(
            f"La directory di submission contiene file inattesi: {sorted(unexpected)}"
        )

    state_dict = clone_state_dict_to_cpu(final_state_dict)
    assert_finite_state(state_dict)
    artifact_payload = {
        "state_dict": state_dict,
        "architecture": deepcopy(original_payload["architecture"]),
        "best_hyperparameters": {
            **deepcopy(original_payload["best_hyperparameters"]),
            "unlearning_method": str(final_config["method"]),
            "unlearning_config": deepcopy(dict(final_config)),
        },
        "model_class_source": original_payload["model_class_source"],
        "feature_columns": list(feature_columns),
        "target_columns": list(target_columns),
        "unlearning_metadata": {
            "official_mia_replication": False,
            "privacy_metric_type": "local_retrained_reference_proxy",
            "final_metrics": deepcopy(dict(final_metrics)),
            "method": deepcopy(dict(method_metadata)),
            "reproducibility": {
                "seed": int(seed),
                "validation_fraction": float(validation_fraction),
            },
        },
    }
    # Validate the complete in-memory artifact before creating any submission file.
    # This prevents a shape or schema error from leaving a plausible-looking artifact.
    validate_artifact_payload(artifact_payload)

    artifact_path = destination / "model_artifact"
    execution_path = destination / "execution_time.txt"
    validation_path = destination / "validation_ids.csv"
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    execution_path.write_text(
        str(int(math.ceil(execution_time_seconds))), encoding="utf-8"
    )
    validation_frame.to_csv(validation_path, index=False)
    return {
        "model_artifact": artifact_path,
        "execution_time": execution_path,
        "validation_ids": validation_path,
    }


def validate_submission(
    submission_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    inference_batch_size: int = 8,
) -> dict[str, Any]:
    """Valida struttura, artifact, tempo, ID e inferenza opzionale su dati reali."""
    destination = Path(submission_dir)
    if not destination.is_dir():
        raise FileNotFoundError(f"Directory di submission non trovata: {destination}")
    directory_entries = list(destination.iterdir())
    entries = {entry.name for entry in directory_entries}
    if entries != SUBMISSION_FILENAMES:
        missing = sorted(SUBMISSION_FILENAMES - entries)
        extra = sorted(entries - SUBMISSION_FILENAMES)
        raise ValueError(f"Submission non valida; mancanti={missing}, extra={extra}.")
    non_files = sorted(entry.name for entry in directory_entries if not entry.is_file())
    if non_files:
        raise ValueError(f"Gli elementi della submission devono essere file: {non_files}.")

    artifact_path = destination / "model_artifact"
    if artifact_path.suffix:
        raise ValueError("model_artifact non deve avere estensione.")
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("model_artifact deve contenere un dizionario.")
    model = validate_artifact_payload(payload)
    state_dict = payload["state_dict"]
    non_cpu = [
        name for name, tensor in state_dict.items() if tensor.device.type != "cpu"
    ]
    if non_cpu:
        raise ValueError(f"Tensori non salvati su CPU: {non_cpu}")
    assert_finite_state(state_dict)

    raw_time = (destination / "execution_time.txt").read_text(encoding="utf-8")
    if re.fullmatch(r"\d+", raw_time) is None:
        raise ValueError(
            "execution_time.txt deve contenere un solo intero non negativo."
        )
    declared_time = int(raw_time)

    validation_frame = pd.read_csv(destination / "validation_ids.csv")
    if list(validation_frame.columns) != [ID_COLUMN]:
        raise ValueError("validation_ids.csv deve avere la sola colonna user_id.")
    if validation_frame[ID_COLUMN].isna().any():
        raise ValueError("validation_ids.csv contiene ID mancanti.")
    if validation_frame[ID_COLUMN].duplicated().any():
        raise ValueError("validation_ids.csv contiene ID duplicati.")
    if validation_frame.empty:
        raise ValueError("validation_ids.csv non puo' essere vuoto.")

    inference_checked = False
    if data_dir is not None:
        if inference_batch_size <= 0:
            raise ValueError("inference_batch_size deve essere positivo.")
        reproducibility = payload.get("unlearning_metadata", {}).get(
            "reproducibility", {}
        )
        if not {"seed", "validation_fraction"} <= set(reproducibility):
            raise KeyError("Mancano i metadati necessari a ricostruire lo split.")
        data = load_challenge_data(
            data_dir,
            validation_fraction=float(reproducibility["validation_fraction"]),
            seed=int(reproducibility["seed"]),
        )
        validate_data_model_compatibility(
            data,
            payload["architecture"],
            feature_columns=payload.get("feature_columns"),
            target_columns=payload.get("target_columns"),
        )
        actual_ids = validation_frame[ID_COLUMN].to_numpy()
        if not np.array_equal(actual_ids, data.validation_ids):
            raise ValueError(
                "Gli ID di validation non coincidono con lo split deterministico."
            )
        validation_ids = set(actual_ids.tolist())
        forget_ids = set(data.forget_frame[ID_COLUMN].tolist())
        retain_train_ids = set(data.retain_train_frame[ID_COLUMN].tolist())
        if validation_ids & forget_ids:
            raise ValueError("Validation e forget set si sovrappongono.")
        if validation_ids & retain_train_ids:
            raise ValueError("La validation e' stata inclusa nel retain training.")
        sample_count = min(inference_batch_size, len(data.x_validation))
        logits = predict_logits(
            model,
            data.x_validation[:sample_count],
            device=torch.device("cpu"),
            batch_size=sample_count,
        )
        if logits.shape != (sample_count, len(data.schema.target_columns)):
            raise ValueError("Shape di inferenza non compatibile con le target.")
        inference_checked = True

    return {
        "submission_dir": str(destination),
        "files": sorted(entries),
        "declared_execution_time_seconds": declared_time,
        "validation_id_count": int(len(validation_frame)),
        "inference_checked": inference_checked,
    }
