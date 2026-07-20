"""Training deterministico per baseline e retraining finale."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import evaluate_model
from .model import model_state_to_cpu


@dataclass
class TrainingResult:
    """Risultato di un training con checkpoint selezionato."""

    model: torch.nn.Module
    best_state_dict: dict[str, torch.Tensor]
    best_epoch: int
    best_precision_at_10: float
    elapsed_seconds: float
    history: pd.DataFrame


def seed_everything(seed: int) -> None:
    """Imposta i seed di Python, NumPy e PyTorch su CPU/GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_training_batch_size(sample_count: int, requested: int) -> int:
    """Evita batch finali di un solo esempio con modelli BatchNorm1d."""
    if sample_count < 2:
        raise ValueError("Il training con BatchNorm richiede almeno due esempi.")
    batch_size = min(max(2, requested), sample_count)
    if sample_count % batch_size == 1:
        if batch_size > 2:
            batch_size -= 1
        else:
            batch_size = sample_count
    return batch_size


def make_data_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    batchnorm_training: bool = False,
) -> DataLoader:
    """Costruisce un DataLoader deterministico mantenendo allineati gli array."""
    if not arrays or any(len(array) != len(arrays[0]) for array in arrays):
        raise ValueError(
            "Gli array del DataLoader devono essere non vuoti e allineati."
        )
    effective_batch_size = (
        _safe_training_batch_size(len(arrays[0]), batch_size)
        if batchnorm_training
        else min(batch_size, len(arrays[0]))
    )
    dataset = TensorDataset(
        *(torch.as_tensor(array, dtype=torch.float32) for array in arrays)
    )
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def compute_positive_class_weights(
    targets: np.ndarray,
    *,
    device: torch.device,
    minimum: float = 0.1,
    maximum: float = 100.0,
) -> torch.Tensor:
    """Calcola pesi positivi multilabel con clipping esplicito."""
    positive_counts = targets.sum(axis=0)
    if np.any(positive_counts <= 0):
        missing = np.flatnonzero(positive_counts <= 0).tolist()
        raise ValueError(f"Target senza esempi positivi agli indici: {missing}")
    negative_counts = len(targets) - positive_counts
    weights = np.clip(negative_counts / positive_counts, minimum, maximum)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def build_optimizer(
    model: torch.nn.Module,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.0,
) -> torch.optim.Optimizer:
    """Costruisce uno degli optimizer supportati dalla configurazione."""
    normalized = name.lower()
    if normalized == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if normalized == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if normalized == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise ValueError(f"Optimizer non supportato: {name!r}.")


def train_fixed_epochs(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    positive_class_weights: torch.Tensor,
    momentum: float = 0.0,
    gradient_clip: float | None = None,
) -> pd.DataFrame:
    """Addestra per un numero fisso di epoche senza consultare la validation."""
    if epochs < 0:
        raise ValueError("epochs non puo' essere negativo.")
    seed_everything(seed)
    loader = make_data_loader(
        features,
        targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        batchnorm_training=True,
    )
    optimizer = build_optimizer(
        model,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    class_weights = positive_class_weights.to(device)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_loss = 0.0
        processed = 0
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=device.type == "cuda")
            target_batch = target_batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(feature_batch), target_batch, pos_weight=class_weights
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Loss non finita all'epoca {epoch}.")
            loss.backward()
            if gradient_clip is not None:
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise FloatingPointError("Norma dei gradienti non finita.")
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * len(feature_batch)
            processed += len(feature_batch)
        history.append(
            {"epoch": epoch, "train_loss": weighted_loss / max(processed, 1)}
        )
    model.eval()
    return pd.DataFrame(history)


def train_with_early_stopping(
    model: torch.nn.Module,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    evaluation_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    positive_class_weights: torch.Tensor,
    momentum: float = 0.0,
) -> TrainingResult:
    """Addestra una baseline e seleziona il checkpoint su Precision@10.

    Questa funzione appartiene esclusivamente alla ricerca. Il workflow finale
    usa `train_fixed_epochs` e non consulta la validation durante il timer.
    """
    seed_everything(seed)
    loader = make_data_loader(
        train_features,
        train_targets,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        batchnorm_training=True,
    )
    optimizer = build_optimizer(
        model,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
    )
    class_weights = positive_class_weights.to(device)
    initial = evaluate_model(
        model,
        validation_features,
        validation_targets,
        device=device,
        batch_size=evaluation_batch_size,
    )
    best_precision = float(initial["precision_at_10"])
    best_state = model_state_to_cpu(model)
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        weighted_loss = 0.0
        processed = 0
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, non_blocking=device.type == "cuda")
            target_batch = target_batch.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(feature_batch), target_batch, pos_weight=class_weights
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Loss non finita all'epoca {epoch}.")
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().item()) * len(feature_batch)
            processed += len(feature_batch)

        metrics = evaluate_model(
            model,
            validation_features,
            validation_targets,
            device=device,
            batch_size=evaluation_batch_size,
        )
        precision = float(metrics["precision_at_10"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": weighted_loss / max(processed, 1),
                "validation_bce": float(metrics["bce_from_logits"]),
                "validation_precision_at_10": precision,
            }
        )
        if precision > best_precision:
            best_precision = precision
            best_state = model_state_to_cpu(model)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return TrainingResult(
        model=model,
        best_state_dict=best_state,
        best_epoch=best_epoch,
        best_precision_at_10=best_precision,
        elapsed_seconds=float(elapsed),
        history=pd.DataFrame(history),
    )
