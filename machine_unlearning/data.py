"""Caricamento, validazione e preprocessing dei dati della challenge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ID_COLUMN = "user_id"
TARGET_PREFIX = "target__"
DEFAULT_TRAIN_PATTERN = "*c000.csv"


@dataclass(frozen=True)
class DatasetSchema:
    """Descrive l'ordine autorevole di identificativo, feature e target."""

    id_column: str
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]


@dataclass
class ChallengeData:
    """Contiene frame originali e array preprocessati per tutti gli split."""

    schema: DatasetSchema
    full_frame: pd.DataFrame
    retain_frame: pd.DataFrame
    retain_train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    forget_frame: pd.DataFrame
    x_retain_train: np.ndarray
    y_retain_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_forget: np.ndarray
    y_forget: np.ndarray
    replaced_feature_values: dict[str, int]
    seed: int
    validation_fraction: float

    @property
    def validation_ids(self) -> np.ndarray:
        """Restituisce gli ID di validation nell'ordine dello split."""
        return self.validation_frame[self.schema.id_column].to_numpy(copy=True)


def _read_csv(path: Path, *, expected_separator: str | None = None) -> pd.DataFrame:
    """Legge un CSV usando il separatore atteso o quello rilevato dall'header."""
    if expected_separator is not None:
        return pd.read_csv(path, sep=expected_separator)

    with path.open("r", encoding="utf-8-sig") as handle:
        header = handle.readline()
    separator = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=separator)


def _validate_unique_ids(frame: pd.DataFrame, *, name: str, id_column: str) -> None:
    """Rifiuta ID mancanti o duplicati, che renderebbero ambiguo lo split."""
    if id_column not in frame.columns:
        raise ValueError(f"Colonna {id_column!r} assente in {name}.")
    if frame[id_column].isna().any():
        raise ValueError(f"{name} contiene {id_column} mancanti.")
    duplicate_count = int(frame[id_column].duplicated().sum())
    if duplicate_count:
        raise ValueError(
            f"{name} contiene {duplicate_count} {id_column} duplicati; "
            "la challenge richiede split per utente univoco."
        )


def preprocess_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, int]:
    """Converte le feature in float32 e sostituisce valori non finiti con zero.

    Manteniamo il preprocessing usato dal modello originale: conversione
    numerica seguita da zero filling. Il secondo valore restituito rende
    osservabile quante celle sono state sostituite.
    """
    numeric = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    replaced = int(numeric.isna().sum().sum())
    values = numeric.fillna(0.0).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Il preprocessing ha prodotto feature non finite.")
    return values, replaced


def preprocess_targets(
    frame: pd.DataFrame,
    target_columns: Sequence[str],
) -> np.ndarray:
    """Converte le target in float32 senza imputare valori mancanti.

    Target mancanti, infinite o non binarie indicano dati non validi e vengono
    rifiutate: imputarle cambierebbe silenziosamente la ground truth.
    """
    numeric = frame.loc[:, target_columns].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Le target contengono valori mancanti o non finiti.")
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("Le target devono contenere esclusivamente 0 e 1.")
    return values


def load_challenge_data(
    data_dir: str | Path,
    *,
    validation_fraction: float,
    seed: int,
    train_pattern: str = DEFAULT_TRAIN_PATTERN,
    id_column: str = ID_COLUMN,
    target_prefix: str = TARGET_PREFIX,
) -> ChallengeData:
    """Carica i dati e costruisce split deterministici e disgiunti.

    Le partizioni di training vengono concatenate nell'ordine lessicografico dei
    nomi. Il forget set viene verificato contro il training completo; poi
    separiamo il retain in train e validation usando il seed fornito.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction deve essere compresa tra 0 e 1.")

    data_path = Path(data_dir)
    train_paths = sorted(data_path.glob(train_pattern))
    forget_path = data_path / "forget_data.csv"
    if not train_paths:
        raise FileNotFoundError(
            f"Nessuna partizione trovata in {data_path} con pattern {train_pattern!r}."
        )
    if not forget_path.is_file():
        raise FileNotFoundError(f"Forget set non trovato: {forget_path}")

    training_parts = [_read_csv(path, expected_separator=";") for path in train_paths]
    authoritative_columns = list(training_parts[0].columns)
    for path, part in zip(train_paths, training_parts):
        if list(part.columns) != authoritative_columns:
            raise ValueError(f"Schema o ordine colonne incoerente in {path.name}.")

    full_frame = pd.concat(training_parts, ignore_index=True)
    forget_frame = _read_csv(forget_path)
    if set(forget_frame.columns) != set(authoritative_columns):
        missing = sorted(set(authoritative_columns) - set(forget_frame.columns))
        extra = sorted(set(forget_frame.columns) - set(authoritative_columns))
        raise ValueError(
            f"Schema forget incompatibile; colonne mancanti={missing}, extra={extra}."
        )
    forget_frame = forget_frame.loc[:, authoritative_columns].copy()

    _validate_unique_ids(full_frame, name="training completo", id_column=id_column)
    _validate_unique_ids(forget_frame, name="forget set", id_column=id_column)

    target_columns = tuple(
        column
        for column in authoritative_columns
        if column.lower().startswith(target_prefix.lower())
    )
    if not target_columns:
        raise ValueError(f"Nessuna target trovata con prefisso {target_prefix!r}.")
    feature_columns = tuple(
        column
        for column in authoritative_columns
        if column != id_column and column not in target_columns
    )
    schema = DatasetSchema(id_column, feature_columns, target_columns)

    training_ids = set(full_frame[id_column].tolist())
    forget_ids = set(forget_frame[id_column].tolist())
    missing_forget_ids = forget_ids - training_ids
    if missing_forget_ids:
        raise ValueError(
            f"{len(missing_forget_ids)} ID forget non sono presenti nel training."
        )

    retain_frame = full_frame.loc[~full_frame[id_column].isin(forget_ids)].copy()
    retain_train_frame, validation_frame = train_test_split(
        retain_frame,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    retain_train_frame = retain_train_frame.copy()
    validation_frame = validation_frame.copy()

    retain_train_ids = set(retain_train_frame[id_column].tolist())
    validation_ids = set(validation_frame[id_column].tolist())
    if retain_train_ids & validation_ids:
        raise RuntimeError("Retain train e validation si sovrappongono.")
    if retain_train_ids & forget_ids:
        raise RuntimeError("Il retain train contiene utenti del forget set.")
    if validation_ids & forget_ids:
        raise RuntimeError("Validation e forget set si sovrappongono.")
    if len(retain_train_ids | validation_ids | forget_ids) != len(full_frame):
        raise RuntimeError("Gli split non ricostruiscono il training completo.")

    x_retain, replaced_retain = preprocess_features(retain_train_frame, feature_columns)
    x_validation, replaced_validation = preprocess_features(
        validation_frame, feature_columns
    )
    x_forget, replaced_forget = preprocess_features(forget_frame, feature_columns)

    return ChallengeData(
        schema=schema,
        full_frame=full_frame,
        retain_frame=retain_frame,
        retain_train_frame=retain_train_frame,
        validation_frame=validation_frame,
        forget_frame=forget_frame,
        x_retain_train=x_retain,
        y_retain_train=preprocess_targets(retain_train_frame, target_columns),
        x_validation=x_validation,
        y_validation=preprocess_targets(validation_frame, target_columns),
        x_forget=x_forget,
        y_forget=preprocess_targets(forget_frame, target_columns),
        replaced_feature_values={
            "retain_train": replaced_retain,
            "validation": replaced_validation,
            "forget": replaced_forget,
        },
        seed=seed,
        validation_fraction=validation_fraction,
    )


def save_validation_ids(data: ChallengeData, path: str | Path) -> Path:
    """Salva gli ID di validation con l'unico header ammesso dalla challenge."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID_COLUMN: data.validation_ids}).to_csv(destination, index=False)
    return destination


def validate_data_model_compatibility(
    data: ChallengeData,
    architecture: dict[str, object],
) -> None:
    """Controlla che dimensioni di feature e target coincidano con il modello."""
    input_dim = int(architecture["input_dim"])
    output_dim = int(architecture["num_outputs"])
    if len(data.schema.feature_columns) != input_dim:
        raise ValueError(
            f"Il dataset ha {len(data.schema.feature_columns)} feature, "
            f"ma il modello ne attende {input_dim}."
        )
    if len(data.schema.target_columns) != output_dim:
        raise ValueError(
            f"Il dataset ha {len(data.schema.target_columns)} target, "
            f"ma il modello ne attende {output_dim}."
        )
