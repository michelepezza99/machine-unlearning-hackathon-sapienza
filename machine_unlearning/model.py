"""Definizione autorevole del modello e gestione degli artifact."""

from __future__ import annotations

import inspect
import pickle
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
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

    def predict_proba(self, features: np.ndarray | torch.Tensor) -> np.ndarray:
        """Calcola probabilita' sigmoid sul device corrente del modello."""
        self.eval()
        tensor = torch.as_tensor(features, dtype=torch.float32)
        device = next(self.parameters()).device
        with torch.inference_mode():
            probabilities = torch.sigmoid(self(tensor.to(device)))
        return probabilities.cpu().numpy()


def clone_state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clona uno stato su CPU per impedire alias e dipendenze dal device."""
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


def _validated_architecture(payload: Mapping[str, Any]) -> dict[str, Any]:
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise TypeError("La chiave 'architecture' deve contenere un mapping.")
    required = {"input_dim", "hidden_layers", "num_outputs"}
    missing = required - set(architecture)
    if missing:
        raise KeyError(f"Architettura incompleta: {sorted(missing)}")
    hidden_layers = [int(value) for value in architecture["hidden_layers"]]
    normalized = {
        "input_dim": int(architecture["input_dim"]),
        "hidden_layers": hidden_layers,
        "num_outputs": int(architecture["num_outputs"]),
    }
    if normalized["input_dim"] <= 0 or normalized["num_outputs"] <= 0:
        raise ValueError("Le dimensioni del modello devono essere positive.")
    if any(width <= 0 for width in hidden_layers):
        raise ValueError("Tutti gli hidden layer devono avere ampiezza positiva.")
    return normalized


def validate_artifact_payload(payload: Mapping[str, Any]) -> DynamicMLP:
    """Valida schema e caricamento stretto di un artifact su CPU."""
    missing = REQUIRED_ARTIFACT_KEYS - set(payload)
    if missing:
        raise KeyError(f"Artifact incompleto: {sorted(missing)}")
    if not isinstance(payload["state_dict"], Mapping):
        raise TypeError("La chiave 'state_dict' deve contenere un mapping.")

    architecture = _validated_architecture(payload)
    state_dict = clone_state_dict_to_cpu(payload["state_dict"])
    assert_finite_state(state_dict)
    model = DynamicMLP(**architecture)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_model_artifact(path: str | Path) -> dict[str, Any]:
    """Carica un artifact, ne valida lo schema e normalizza i tensori su CPU."""
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact non trovato: {artifact_path}")
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Il model artifact deve contenere un dizionario.")
    validate_artifact_payload(payload)
    normalized = deepcopy(payload)
    normalized["architecture"] = _validated_architecture(payload)
    normalized["state_dict"] = clone_state_dict_to_cpu(payload["state_dict"])
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
