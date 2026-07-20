"""Metriche autorevoli di utility e proxy locali di privacy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ReferencePrivacyProxy:
    """Proxy MIA locale addestrata con predizioni out-of-fold per utente."""

    fold_estimators: list[tuple[np.ndarray, Any]]
    reverse: bool
    cv_auc: float
    original_features: np.ndarray
    retrained_features: np.ndarray
    feature_scale: np.ndarray
    original_membership_mean: float
    retrained_membership_mean: float
    original_reference_distance: float


def predict_logits(
    model: torch.nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Esegue inferenza a batch senza modificare BatchNorm o costruire grafi."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("Le feature da valutare devono essere una matrice non vuota.")
    model.eval()
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.as_tensor(
                features[start : start + batch_size], dtype=torch.float32, device=device
            )
            batches.append(model(batch).detach().cpu())
    logits = torch.cat(batches, dim=0).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Il modello ha prodotto logit non finiti.")
    return logits


def precision_at_k(logits: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Calcola la Precision@k media per classificazione multilabel."""
    if logits.ndim != 2 or targets.ndim != 2 or logits.shape != targets.shape:
        raise ValueError(f"Shape incompatibili: {logits.shape} e {targets.shape}.")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError(f"k={k} non valido per {logits.shape[1]} output.")
    if not np.isfinite(logits).all() or not np.isfinite(targets).all():
        raise ValueError("Precision@k non accetta valori non finiti.")
    top_indices = np.argpartition(logits, logits.shape[1] - k, axis=1)[:, -k:]
    hits = np.take_along_axis(targets, top_indices, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def binary_cross_entropy_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Calcola la BCE media direttamente sui logit."""
    if logits.ndim != 2 or targets.ndim != 2 or logits.shape != targets.shape:
        raise ValueError(f"Shape incompatibili: {logits.shape} e {targets.shape}.")
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(
        logits_tensor, targets_tensor, reduction="mean"
    )
    if not bool(torch.isfinite(loss)):
        raise ValueError("La BCE non e' finita.")
    return float(loss.item())


def _attack_features(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Costruisce segnali confidence/loss-based per la sola proxy locale."""
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    targets_tensor = torch.as_tensor(targets, dtype=torch.float32)
    probabilities = torch.sigmoid(logits_tensor).clamp(1e-7, 1.0 - 1e-7)
    sample_loss = F.binary_cross_entropy_with_logits(
        logits_tensor, targets_tensor, reduction="none"
    ).mean(dim=1)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    ).mean(dim=1)
    confidence = torch.maximum(probabilities, 1.0 - probabilities).mean(dim=1)
    top_k_mean = probabilities.topk(min(10, probabilities.shape[1]), dim=1).values.mean(
        dim=1
    )
    top_two = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
    margin = (
        top_two[:, 0] - top_two[:, 1] if probabilities.shape[1] >= 2 else top_two[:, 0]
    )
    logit_norm = logits_tensor.square().mean(dim=1).sqrt()
    positive_confidence = (probabilities * targets_tensor).sum(
        dim=1
    ) / targets_tensor.sum(dim=1).clamp_min(1.0)
    negative_targets = 1.0 - targets_tensor
    negative_confidence = ((1.0 - probabilities) * negative_targets).sum(
        dim=1
    ) / negative_targets.sum(dim=1).clamp_min(1.0)
    return (
        torch.stack(
            [
                sample_loss,
                entropy,
                confidence,
                top_k_mean,
                margin,
                logit_norm,
                positive_confidence,
                negative_confidence,
            ],
            dim=1,
        )
        .numpy()
        .astype(np.float64, copy=False)
    )


def fit_reference_privacy_proxy(
    original_forget_logits: np.ndarray,
    retrained_forget_logits: np.ndarray,
    forget_targets: np.ndarray,
    *,
    seed: int,
) -> ReferencePrivacyProxy:
    """Addestra una proxy locale che confronta originale e retrained.

    Raggruppiamo le due osservazioni dello stesso utente nello stesso fold. Gli
    estimator conservati valutano poi ogni candidato soltanto sugli utenti che
    non hanno visto in addestramento, evitando leakage fra coppie correlate.
    """
    if original_forget_logits.shape != retrained_forget_logits.shape:
        raise ValueError("I logit originale e retrained devono essere allineati.")
    original_features = _attack_features(original_forget_logits, forget_targets)
    retrained_features = _attack_features(retrained_forget_logits, forget_targets)
    sample_count = len(forget_targets)
    if sample_count < 4:
        raise ValueError("Servono almeno quattro utenti per la proxy privacy.")

    attack_x = np.vstack([original_features, retrained_features])
    labels = np.concatenate(
        [np.ones(sample_count, dtype=int), np.zeros(sample_count, dtype=int)]
    )
    groups = np.concatenate([np.arange(sample_count), np.arange(sample_count)])
    out_of_fold = np.zeros(len(labels), dtype=np.float64)
    fold_estimators: list[tuple[np.ndarray, Any]] = []

    splitter = GroupKFold(n_splits=min(5, sample_count))
    for train_indices, test_indices in splitter.split(attack_x, labels, groups):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            ),
        )
        estimator.fit(attack_x[train_indices], labels[train_indices])
        out_of_fold[test_indices] = estimator.predict_proba(attack_x[test_indices])[
            :, 1
        ]
        candidate_indices = np.unique(groups[test_indices]).astype(np.int64)
        fold_estimators.append((candidate_indices, estimator))

    original_membership = out_of_fold[:sample_count]
    retrained_membership = out_of_fold[sample_count:]
    reverse = bool(original_membership.mean() < retrained_membership.mean())
    if reverse:
        original_membership = 1.0 - original_membership
        retrained_membership = 1.0 - retrained_membership

    feature_scale = np.std(retrained_features, axis=0, ddof=1)
    fallback_scale = np.std(
        np.vstack([original_features, retrained_features]), axis=0, ddof=1
    )
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, fallback_scale)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    original_distance = float(
        np.mean(np.abs(original_features - retrained_features) / feature_scale)
    )

    return ReferencePrivacyProxy(
        fold_estimators=fold_estimators,
        reverse=reverse,
        cv_auc=float(roc_auc_score(labels, out_of_fold)),
        original_features=original_features,
        retrained_features=retrained_features,
        feature_scale=feature_scale,
        original_membership_mean=float(original_membership.mean()),
        retrained_membership_mean=float(retrained_membership.mean()),
        original_reference_distance=max(original_distance, 1e-12),
    )


def reference_privacy_metrics(
    candidate_forget_logits: np.ndarray,
    forget_targets: np.ndarray,
    proxy: ReferencePrivacyProxy,
) -> dict[str, float]:
    """Confronta un candidato col retrained usando la proxy out-of-fold."""
    candidate_features = _attack_features(candidate_forget_logits, forget_targets)
    membership = np.full(len(candidate_features), np.nan, dtype=np.float64)
    for candidate_indices, estimator in proxy.fold_estimators:
        membership[candidate_indices] = estimator.predict_proba(
            candidate_features[candidate_indices]
        )[:, 1]
    if np.isnan(membership).any():
        raise RuntimeError("La proxy non ha prodotto una stima per ogni utente.")
    if proxy.reverse:
        membership = 1.0 - membership

    denominator = max(
        proxy.original_membership_mean - proxy.retrained_membership_mean,
        1e-12,
    )
    attack_privacy = float(
        np.clip(
            (proxy.original_membership_mean - membership.mean()) / denominator,
            0.0,
            1.0,
        )
    )
    candidate_distance = float(
        np.mean(
            np.abs(candidate_features - proxy.retrained_features) / proxy.feature_scale
        )
    )
    feature_privacy = float(
        np.clip(
            1.0 - candidate_distance / proxy.original_reference_distance,
            0.0,
            1.0,
        )
    )
    return {
        "local_proxy_membership_mean": float(membership.mean()),
        "local_proxy_attack_privacy": attack_privacy,
        "local_proxy_feature_privacy": feature_privacy,
        "local_privacy_proxy": 0.5 * attack_privacy + 0.5 * feature_privacy,
        "local_proxy_reference_distance": candidate_distance,
    }


def evaluate_model(
    model: torch.nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Calcola logit, Precision@10 e BCE con una sola inferenza."""
    logits = predict_logits(model, features, device=device, batch_size=batch_size)
    return {
        "logits": logits,
        "precision_at_10": precision_at_k(logits, targets, k=10),
        "bce_from_logits": binary_cross_entropy_from_logits(logits, targets),
    }


def evaluate_unlearning_candidate(
    model: torch.nn.Module,
    *,
    validation_features: np.ndarray,
    validation_targets: np.ndarray,
    forget_features: np.ndarray,
    forget_targets: np.ndarray,
    device: torch.device,
    privacy_proxy: ReferencePrivacyProxy,
    baseline_precision_at_10: float,
    retraining_time_seconds: float,
    execution_time_seconds: float,
    batch_size: int = 2048,
) -> dict[str, float]:
    """Valuta utility e proxy privacy di una configurazione sperimentale.

    Il punteggio restituito serve esclusivamente alla ricerca locale: la parte
    privacy non replica la MIA ufficiale nascosta della challenge.
    """
    validation = evaluate_model(
        model,
        validation_features,
        validation_targets,
        device=device,
        batch_size=batch_size,
    )
    forget = evaluate_model(
        model,
        forget_features,
        forget_targets,
        device=device,
        batch_size=batch_size,
    )
    privacy = reference_privacy_metrics(forget["logits"], forget_targets, privacy_proxy)
    local_score, time_proxy = local_multi_objective_score(
        precision_at_10_value=float(validation["precision_at_10"]),
        local_privacy_proxy=privacy["local_privacy_proxy"],
        execution_time_seconds=execution_time_seconds,
        retraining_time_seconds=retraining_time_seconds,
    )
    return {
        "precision_at_10": float(validation["precision_at_10"]),
        "validation_bce": float(validation["bce_from_logits"]),
        "forget_bce": float(forget["bce_from_logits"]),
        "utility_ratio": float(
            validation["precision_at_10"] / max(baseline_precision_at_10, 1e-12)
        ),
        "local_time_proxy": time_proxy,
        "local_search_score": local_score,
        **privacy,
    }


def local_multi_objective_score(
    *,
    precision_at_10_value: float,
    local_privacy_proxy: float,
    execution_time_seconds: float,
    retraining_time_seconds: float,
) -> tuple[float, float]:
    """Combina utility, proxy privacy e tempo per la sola ricerca locale."""
    time_reference = max(float(retraining_time_seconds), 1.0)
    time_proxy = float(math.exp(-max(execution_time_seconds, 0.0) / time_reference))
    score = (
        0.45 * precision_at_10_value + 0.45 * local_privacy_proxy + 0.10 * time_proxy
    )
    return float(score), time_proxy
