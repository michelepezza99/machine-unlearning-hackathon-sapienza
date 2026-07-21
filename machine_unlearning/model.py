"""Definizione autorevole del modello e gestione degli artifact."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any

import torch
from torch import nn


REQUIRED_ARTIFACT_KEYS = {
    "state_dict",
    "architecture",
    "best_hyperparameters",
    "model_class_source",
}


class DynamicMLP(nn.Module):
    """MLP dinamico compatibile con lo `state_dict` fornito dalla challenge."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: list[int] | tuple[int, ...],
        num_outputs: int,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, num_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Restituisce i logit multilabel."""
        return self.net(features)


def clone_state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clona uno stato su CPU per impedire alias e dipendenze dal device."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise TypeError("Lo state_dict deve essere un mapping non vuoto.")
    invalid_names = [
        name for name in state_dict if not isinstance(name, str) or not name
    ]
    if invalid_names:
        raise TypeError(
            "Ogni chiave dello state_dict deve essere una stringa non vuota."
        )
    invalid_values = [
        name
        for name, tensor in state_dict.items()
        if not isinstance(tensor, torch.Tensor)
    ]
    if invalid_values:
        raise TypeError(
            "Ogni valore dello state_dict deve essere un tensore; "
            f"valori non validi: {invalid_values}."
        )
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def model_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copia su CPU parametri e buffer di un modello."""
    return clone_state_dict_to_cpu(model.state_dict())


def assert_finite_state(state_dict: Mapping[str, torch.Tensor]) -> None:
    """Rifiuta stati contenenti NaN o infinito prima della serializzazione."""
    non_finite = [
        name
        for name, tensor in state_dict.items()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all())
    ]
    if non_finite:
        raise ValueError(f"Tensori non finiti nello state_dict: {non_finite}")


def _positive_integer(value: Any, *, field_name: str) -> int:
    """Convalida dimensioni intere senza accettare bool o conversioni con perdita."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} deve essere un intero positivo.")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} deve essere positivo.")
    return normalized


def _validated_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("La chiave 'architecture' deve contenere un mapping.")
    required = {"input_dim", "hidden_layers", "num_outputs"}
    missing = required - set(architecture)
    if missing:
        raise KeyError(f"Architettura incompleta: {sorted(missing)}")
    raw_hidden_layers = architecture["hidden_layers"]
    if not isinstance(raw_hidden_layers, (list, tuple)):
        raise TypeError("architecture.hidden_layers deve essere una lista o tupla.")
    hidden_layers = [
        _positive_integer(value, field_name=f"hidden_layers[{index}]")
        for index, value in enumerate(raw_hidden_layers)
    ]
    normalized = {
        "input_dim": _positive_integer(
            architecture["input_dim"], field_name="architecture.input_dim"
        ),
        "hidden_layers": hidden_layers,
        "num_outputs": _positive_integer(
            architecture["num_outputs"], field_name="architecture.num_outputs"
        ),
    }
    return normalized


def _validated_column_names(
    payload: Mapping[str, Any],
    *,
    key: str,
    expected_count: int,
) -> list[str] | None:
    """Convalida gli ordini colonna opzionali aggiunti agli artifact finali."""
    if key not in payload:
        return None
    raw_names = payload[key]
    if not isinstance(raw_names, (list, tuple)):
        raise TypeError(f"La chiave {key!r} deve contenere una lista di nomi.")
    names = list(raw_names)
    if len(names) != expected_count:
        raise ValueError(
            f"La chiave {key!r} contiene {len(names)} nomi, "
            f"ma il modello ne richiede {expected_count}."
        )
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"La chiave {key!r} contiene un nome non valido.")
    if len(set(names)) != len(names):
        raise ValueError(f"La chiave {key!r} contiene nomi duplicati.")
    return names


def _normalize_artifact_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Valida e copia un payload senza duplicare inutilmente i tensori del modello."""
    if not isinstance(payload, Mapping):
        raise TypeError("Il model artifact deve contenere un mapping.")
    missing = REQUIRED_ARTIFACT_KEYS - set(payload)
    if missing:
        raise KeyError(f"Artifact incompleto: {sorted(missing)}")
    if not isinstance(payload["best_hyperparameters"], Mapping):
        raise TypeError("La chiave 'best_hyperparameters' deve contenere un mapping.")
    if not isinstance(payload["model_class_source"], str) or not payload[
        "model_class_source"
    ].strip():
        raise TypeError(
            "La chiave 'model_class_source' deve essere una stringa non vuota."
        )

    architecture = _validated_architecture(payload)
    state_dict = clone_state_dict_to_cpu(payload["state_dict"])
    assert_finite_state(state_dict)
    feature_columns = _validated_column_names(
        payload,
        key="feature_columns",
        expected_count=architecture["input_dim"],
    )
    target_columns = _validated_column_names(
        payload,
        key="target_columns",
        expected_count=architecture["num_outputs"],
    )

    # Extra diagnostics are small JSON-like objects. Excluding state_dict here
    # avoids a second deep tensor copy during artifact loading.
    normalized = deepcopy(
        {key: value for key, value in payload.items() if key != "state_dict"}
    )
    normalized["architecture"] = architecture
    normalized["best_hyperparameters"] = deepcopy(dict(payload["best_hyperparameters"]))
    normalized["model_class_source"] = payload["model_class_source"]
    normalized["state_dict"] = state_dict
    if feature_columns is not None:
        normalized["feature_columns"] = feature_columns
    if target_columns is not None:
        normalized["target_columns"] = target_columns
    return normalized


def _model_from_normalized_payload(payload: Mapping[str, Any]) -> DynamicMLP:
    """Ricostruisce in modo stretto un payload gia' normalizzato."""
    model = DynamicMLP(**payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def validate_artifact_payload(payload: Mapping[str, Any]) -> DynamicMLP:
    """Valida schema e caricamento stretto di un artifact su CPU."""
    normalized = _normalize_artifact_payload(payload)
    return _model_from_normalized_payload(normalized)


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Carica un artifact, ne valida lo schema e normalizza i tensori su CPU."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact non trovato: {artifact_path}")
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    normalized = _normalize_artifact_payload(payload)
    _model_from_normalized_payload(normalized)
    return normalized


def build_model(
    architecture: Mapping[str, Any],
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    device: str | torch.device = "cpu",
) -> DynamicMLP:
    """Costruisce `DynamicMLP` e, se fornito, carica lo stato con `strict=True`."""
    normalized = _validated_architecture({"architecture": architecture})
    model = DynamicMLP(**normalized)
    if state_dict is not None:
        model.load_state_dict(clone_state_dict_to_cpu(state_dict), strict=True)
    model.to(torch.device(device))
    return model


def build_model_from_artifact(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cpu",
) -> DynamicMLP:
    """Ricostruisce il modello originale da un payload gia' validato."""
    model = build_model(
        payload["architecture"],
        state_dict=payload["state_dict"],
        device=device,
    )
    model.eval()
    return model


def dynamic_mlp_source() -> str:
    """Restituisce il sorgente della classe per artifact sintetici e test."""
    return inspect.getsource(DynamicMLP)
