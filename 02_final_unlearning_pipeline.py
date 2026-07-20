from __future__ import annotations

"""Pipeline finale per la TIM x Sapienza Machine Unlearning Hackathon.

Il modulo contiene la parte riutilizzabile della soluzione finale: calcolo della
Fisher diagonale, selezione dei parametri associati al forget set, dampening
selettivo, repair sul retain set, proxy locale di privacy e creazione della
submission. Le funzioni sono pensate per essere chiamate dal notebook tramite
`final_unlearning_runner.py`.

Nota metodologica: la MIA ufficiale non e' pubblica. Le metriche di privacy qui
implementate sono proxy locali basate sul confronto con il modello retrained.
"""

import gc
import json
import math
import pickle
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Configurazione, riproducibilita' e gestione dello stato
# =============================================================================


def seed_everything(seed: int) -> None:
    """Imposta i seed principali per rendere confrontabili gli esperimenti.

    Parametri
    ----------
    seed:
        Valore usato per Python, NumPy, PyTorch CPU e PyTorch CUDA.

    Effetti collaterali
    -------------------
    Configura cuDNN in modalita' deterministica. Questo puo' ridurre leggermente
    le prestazioni, ma rende piu' affidabile il confronto tra configurazioni.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copia lo stato di un modello su CPU in modo indipendente.

    Usiamo questa funzione per checkpoint, artifact e ripristini tra esperimenti:
    ogni configurazione deve ripartire dai pesi originali, non dallo stato lasciato
    da una configurazione precedente.
    """
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def clone_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clona uno `state_dict` gia' disponibile, mantenendo i tensori su CPU."""
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def clear_memory(*objects: Any) -> None:
    """Libera riferimenti temporanei e svuota la cache CUDA quando disponibile.

    La ricerca prova piu' modelli consecutivi. Pulire esplicitamente riduce il
    rischio di saturare la memoria GPU durante Fisher, dampening e repair.
    """
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_indices(n: int, sample_size: int | None, seed: int) -> np.ndarray:
    """Restituisce un sottoinsieme riproducibile di indici ordinati."""
    if sample_size is None or sample_size >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=sample_size, replace=False)).astype(np.int64)


def _make_loader(
    *arrays: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
    drop_last: bool = False,
) -> DataLoader:
    """Costruisce un DataLoader NumPy -> PyTorch con gestione CPU/GPU coerente.

    Il parametro `drop_last` viene controllato dai chiamanti: in training puo'
    essere utile evitare batch di dimensione 1 quando il modello contiene
    BatchNorm1d, mentre in valutazione non vogliamo perdere esempi.
    """
    tensors = [torch.as_tensor(array, dtype=torch.float32) for array in arrays]
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


# =============================================================================
# Predizione, metriche di utility e proxy privacy
# =============================================================================


def predict_logits_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Calcola i logit di un modello e li restituisce come array NumPy.

    Il modello viene messo in `eval()` e l'inferenza usa `torch.inference_mode()`.
    Questo evita gradienti inutili e impedisce a BatchNorm/Dropout di cambiare
    comportamento rispetto alla valutazione.
    """
    model.eval()
    loader = _make_loader(
        X,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        device=device,
    )
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            outputs.append(model(xb).detach().cpu())
    if not outputs:
        raise ValueError("Nessun esempio da valutare.")
    logits = torch.cat(outputs, dim=0).numpy()
    if not np.isfinite(logits).all():
        raise ValueError("Il modello ha prodotto logit non finiti.")
    return logits


def precision_at_k(logits: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
    """Calcola la Precision@k media per un problema multilabel.

    Per ogni utente selezioniamo i `k` logit piu' alti e misuriamo quante target
    positive sono presenti tra questi indici. La challenge usa Precision@10 come
    componente principale dell'utility.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"Shape incompatibili: {logits.shape} vs {targets.shape}.")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError(f"k={k} non valido per {logits.shape[1]} output.")
    topk = np.argpartition(logits, logits.shape[1] - k, axis=1)[:, -k:]
    hits = np.take_along_axis(targets, topk, axis=1).sum(axis=1)
    return float(np.mean(hits / k))


def mean_bce(logits: np.ndarray, targets: np.ndarray) -> float:
    """Restituisce la BCE multilabel media per esempio.

    La funzione opera sui logit, quindi usa `binary_cross_entropy_with_logits`
    senza applicare prima la sigmoid.
    """
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    losses = F.binary_cross_entropy_with_logits(logits_t, targets_t, reduction="none")
    return float(losses.mean(dim=1).mean().item())


def attack_features(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Estrae feature usate spesso da attacchi MIA confidence/loss-based.

    Non stiamo replicando l'attacco ufficiale: costruiamo una proxy locale che
    confronta confidenza, loss, entropia e statistiche dei logit sul forget set.
    """
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    targets_t = torch.as_tensor(targets, dtype=torch.float32)
    probs = torch.sigmoid(logits_t).clamp(1e-7, 1.0 - 1e-7)

    sample_loss = F.binary_cross_entropy_with_logits(
        logits_t, targets_t, reduction="none"
    ).mean(dim=1)
    entropy = -(
        probs * probs.log() + (1.0 - probs) * (1.0 - probs).log()
    ).mean(dim=1)
    max_conf = torch.maximum(probs, 1.0 - probs).mean(dim=1)
    topk = min(10, probs.shape[1])
    topk_mean = probs.topk(topk, dim=1).values.mean(dim=1)
    top2 = probs.topk(min(2, probs.shape[1]), dim=1).values
    margin = top2[:, 0] - top2[:, 1] if probs.shape[1] >= 2 else top2[:, 0]
    logit_norm = logits_t.square().mean(dim=1).sqrt()
    positive_conf = (probs * targets_t).sum(dim=1) / targets_t.sum(dim=1).clamp_min(1.0)
    negative_conf = ((1.0 - probs) * (1.0 - targets_t)).sum(dim=1) / (
        1.0 - targets_t
    ).sum(dim=1).clamp_min(1.0)

    features = torch.stack(
        [
            sample_loss,
            entropy,
            max_conf,
            topk_mean,
            margin,
            logit_norm,
            positive_conf,
            negative_conf,
        ],
        dim=1,
    )
    return features.numpy().astype(np.float64, copy=False)


def fit_reference_attack_proxy(
    original_forget_logits: np.ndarray,
    retrained_forget_logits: np.ndarray,
    y_forget: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """
    Addestra una proxy MIA che distingue comportamento originale e retrained.

    Usiamo gli stessi esempi forget con due modelli diversi: originale e
    retrained. La GroupKFold evita che lo stesso utente appaia sia nel train sia
    nel test della proxy. Questa e' volutamente una proxy basata sul riferimento
    retrained, non la MIA ufficiale nascosta.
    """
    original_features = attack_features(original_forget_logits, y_forget)
    retrained_features = attack_features(retrained_forget_logits, y_forget)
    n = len(y_forget)
    if n < 4:
        raise ValueError("Il forget set è troppo piccolo per costruire la proxy MIA.")

    X_attack = np.vstack([original_features, retrained_features])
    labels = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    groups = np.concatenate([np.arange(n), np.arange(n)])

    n_splits = min(5, n)
    splitter = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(labels), dtype=np.float64)

    for train_idx, test_idx in splitter.split(X_attack, labels, groups):
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=3000,
                random_state=seed,
            ),
        )
        estimator.fit(X_attack[train_idx], labels[train_idx])
        oof[test_idx] = estimator.predict_proba(X_attack[test_idx])[:, 1]

    cv_auc = float(roc_auc_score(labels, oof))
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
        ),
    )
    estimator.fit(X_attack, labels)

    p_original = estimator.predict_proba(original_features)[:, 1]
    p_retrained = estimator.predict_proba(retrained_features)[:, 1]
    reverse = bool(p_original.mean() < p_retrained.mean())
    if reverse:
        p_original = 1.0 - p_original
        p_retrained = 1.0 - p_retrained

    feature_scale = np.std(retrained_features, axis=0, ddof=1)
    fallback = np.std(np.vstack([original_features, retrained_features]), axis=0, ddof=1)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, fallback)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)

    original_distance = float(
        np.mean(np.abs(original_features - retrained_features) / feature_scale)
    )

    return {
        "estimator": estimator,
        "reverse": reverse,
        "cv_auc": cv_auc,
        "original_features": original_features,
        "retrained_features": retrained_features,
        "feature_scale": feature_scale,
        "original_membership_mean": float(p_original.mean()),
        "retrained_membership_mean": float(p_retrained.mean()),
        "original_reference_distance": max(original_distance, 1e-12),
    }


def reference_privacy_metrics(
    candidate_forget_logits: np.ndarray,
    y_forget: np.ndarray,
    attack_proxy: dict[str, Any],
) -> dict[str, float]:
    """Valuta quanto un candidato assomiglia al retrained sul forget set.

    Restituiamo due segnali: probabilita' media di membership stimata dalla proxy
    e distanza normalizzata dalle feature del retrained. La media dei due forma
    `proxy_privacy`, usata solo per scegliere configurazioni locali.
    """
    candidate_features = attack_features(candidate_forget_logits, y_forget)
    estimator = attack_proxy["estimator"]
    p_candidate = estimator.predict_proba(candidate_features)[:, 1]
    if attack_proxy["reverse"]:
        p_candidate = 1.0 - p_candidate

    p_orig = attack_proxy["original_membership_mean"]
    p_retrained = attack_proxy["retrained_membership_mean"]
    denominator = max(p_orig - p_retrained, 1e-12)
    attack_privacy = float(np.clip((p_orig - p_candidate.mean()) / denominator, 0.0, 1.0))

    retrained_features = attack_proxy["retrained_features"]
    scale = attack_proxy["feature_scale"]
    candidate_distance = float(
        np.mean(np.abs(candidate_features - retrained_features) / scale)
    )
    feature_privacy = float(
        np.clip(
            1.0
            - candidate_distance / attack_proxy["original_reference_distance"],
            0.0,
            1.0,
        )
    )

    combined = 0.5 * attack_privacy + 0.5 * feature_privacy
    return {
        "proxy_membership_mean": float(p_candidate.mean()),
        "proxy_attack_privacy": attack_privacy,
        "proxy_feature_privacy": feature_privacy,
        "proxy_privacy": float(combined),
        "proxy_reference_distance": candidate_distance,
    }


def evaluate_candidate(
    model: torch.nn.Module,
    *,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_forget: np.ndarray,
    y_forget: np.ndarray,
    device: torch.device,
    attack_proxy: dict[str, Any],
    baseline_p10: float,
    retraining_time: float,
    execution_time: float,
    batch_size: int = 2048,
) -> dict[str, float]:
    """Valuta un candidato con utility, privacy proxy e score multi-obiettivo.

    Lo score locale rispecchia i pesi pubblici della challenge (45/45/10), ma
    usa una privacy proxy perche' la MIA ufficiale non e' disponibile. Il tempo
    passato alla funzione deve riferirsi al metodo candidato, non alla ricerca.
    """
    val_logits = predict_logits_numpy(
        model, X_validation, device=device, batch_size=batch_size
    )
    forget_logits = predict_logits_numpy(
        model, X_forget, device=device, batch_size=batch_size
    )
    privacy = reference_privacy_metrics(forget_logits, y_forget, attack_proxy)
    p10 = precision_at_k(val_logits, y_validation, k=10)
    time_reference = max(float(retraining_time), 1.0)
    time_score = float(math.exp(-max(execution_time, 0.0) / time_reference))
    local_score = 0.45 * p10 + 0.45 * privacy["proxy_privacy"] + 0.10 * time_score
    return {
        "precision_at_10": p10,
        "validation_bce": mean_bce(val_logits, y_validation),
        "forget_bce": mean_bce(forget_logits, y_forget),
        "utility_ratio": float(p10 / max(baseline_p10, 1e-12)),
        "time_score_proxy": time_score,
        "local_score_proxy": float(local_score),
        **privacy,
    }


# =============================================================================
# Calcolo della Fisher diagonale
# =============================================================================


def _eligible_parameter_names(
    model: torch.nn.Module,
    *,
    include_bias: bool,
    include_batchnorm_affine: bool,
) -> set[str]:
    """Individua i parametri su cui possiamo intervenire.

    Escludiamo buffer e, per default, bias e parametri affine di BatchNorm. I
    running statistics di BatchNorm non sono parametri trainabili e non devono
    entrare nella selezione Fisher.
    """
    batchnorm_parameter_names: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            if module.weight is not None:
                batchnorm_parameter_names.add(prefix + "weight")
            if module.bias is not None:
                batchnorm_parameter_names.add(prefix + "bias")

    eligible: set[str] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not include_bias and parameter.ndim < 2:
            continue
        if not include_batchnorm_affine and name in batchnorm_parameter_names:
            continue
        eligible.add(name)
    return eligible


def empirical_fisher_diagonal(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    device: torch.device,
    sample_size: int | None,
    batch_size: int,
    seed: int,
    pos_weight: torch.Tensor | None = None,
    include_bias: bool = False,
    include_batchnorm_affine: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Calcola la Fisher diagonale empirica esatta.

    La stima e' `mean_i[(d loss_i / d theta)^2]`: accumuliamo il quadrato
    dei gradienti per singolo esempio e normalizziamo sul numero effettivo di
    esempi processati. Il modello resta in `eval()` per usare statistiche
    BatchNorm congelate; aggiornare running mean/variance durante questa fase
    cambierebbe il modello prima ancora dell'unlearning.

    Restituisce la Fisher su CPU e un dizionario di metadati utile per audit e
    presentazione dei risultati.
    """
    if len(X) != len(y):
        raise ValueError("X e y devono avere lo stesso numero di esempi.")
    indices = _sample_indices(len(X), sample_size, seed)
    X_sample = X[indices]
    y_sample = y[indices]
    if len(X_sample) == 0:
        raise ValueError("Campione Fisher vuoto.")

    fisher_model = deepcopy(model).to(device)
    fisher_model.eval()
    eligible = _eligible_parameter_names(
        fisher_model,
        include_bias=include_bias,
        include_batchnorm_affine=include_batchnorm_affine,
    )

    params = {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in fisher_model.named_parameters()
    }
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in fisher_model.named_buffers()
    }
    accumulator = {
        name: torch.zeros_like(parameter, dtype=torch.float64, device=device)
        for name, parameter in params.items()
    }

    if pos_weight is not None:
        pos_weight = pos_weight.detach().to(device=device, dtype=torch.float32)

    def single_sample_loss(
        functional_params: dict[str, torch.Tensor],
        functional_buffers: dict[str, torch.Tensor],
        x_i: torch.Tensor,
        y_i: torch.Tensor,
    ) -> torch.Tensor:
        logits = functional_call(
            fisher_model,
            (functional_params, functional_buffers),
            (x_i.unsqueeze(0),),
        ).squeeze(0)
        return F.binary_cross_entropy_with_logits(
            logits,
            y_i,
            pos_weight=pos_weight,
            reduction="mean",
        )

    per_sample_grad = grad(single_sample_loss)
    batched_grad = vmap(
        per_sample_grad,
        in_dims=(None, None, 0, 0),
        randomness="different",
    )

    loader = _make_loader(
        X_sample,
        y_sample,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        device=device,
    )
    processed = 0
    start = time.perf_counter()

    try:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            sample_grads = batched_grad(params, buffers, xb, yb)
            for name, grad_batch in sample_grads.items():
                if name in eligible:
                    accumulator[name].add_(
                        grad_batch.detach().to(torch.float64).square().sum(dim=0)
                    )
            processed += xb.shape[0]
            del sample_grads, xb, yb
    finally:
        elapsed = time.perf_counter() - start

    if processed != len(X_sample):
        raise RuntimeError(f"Fisher incompleta: {processed}/{len(X_sample)} esempi.")

    full_fisher: dict[str, torch.Tensor] = {}
    for name, tensor in fisher_model.state_dict().items():
        if name in accumulator and name in eligible:
            full_fisher[name] = (accumulator[name] / processed).float().cpu()
        else:
            full_fisher[name] = torch.zeros_like(tensor, dtype=torch.float32).cpu()

    metadata = {
        "sample_size": int(processed),
        "batch_size": int(batch_size),
        "elapsed_seconds": float(elapsed),
        "eligible_parameter_names": sorted(eligible),
        "mode": "eval",
        "normalization": "mean of squared per-example gradients",
        "uses_pos_weight": pos_weight is not None,
    }

    del fisher_model, params, buffers, accumulator
    clear_memory()
    return full_fisher, metadata


# =============================================================================
# Selezione dei parametri e Selective Fisher Dampening
# =============================================================================


def _positive_median(values: torch.Tensor) -> float:
    """Restituisce una scala robusta positiva per normalizzare le Fisher."""
    positive = values[torch.isfinite(values) & (values > 0)]
    if positive.numel() == 0:
        return 1.0
    return max(float(positive.median().item()), 1e-20)


def build_fisher_mask(
    model: torch.nn.Module,
    fisher_retain: dict[str, torch.Tensor],
    fisher_forget: dict[str, torch.Tensor],
    *,
    top_fraction: float,
    forget_absolute_quantile: float,
    epsilon: float = 1e-12,
    include_bias: bool = False,
    include_batchnorm_affine: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    """Seleziona i parametri piu' specifici del forget set.

    Normalizziamo Fisher retain e forget con mediane positive robuste, poi usiamo
    un log-ratio stabilizzato. Un parametro entra nella maschera solo se ha ratio
    elevato e importanza assoluta non trascurabile sul forget set; questo evita
    selezioni spurie prodotte da denominatori quasi nulli.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction deve essere in (0, 1].")
    if not 0 <= forget_absolute_quantile < 1:
        raise ValueError("forget_absolute_quantile deve essere in [0, 1).")

    eligible = _eligible_parameter_names(
        model,
        include_bias=include_bias,
        include_batchnorm_affine=include_batchnorm_affine,
    )

    retain_values = torch.cat(
        [fisher_retain[name].float().reshape(-1) for name in eligible]
    )
    forget_values = torch.cat(
        [fisher_forget[name].float().reshape(-1) for name in eligible]
    )
    retain_scale = _positive_median(retain_values)
    forget_scale = _positive_median(forget_values)

    scores: dict[str, torch.Tensor] = {}
    gates: dict[str, torch.Tensor] = {}
    all_normalized_forget = forget_values / forget_scale
    absolute_threshold = float(
        torch.quantile(all_normalized_forget, forget_absolute_quantile).item()
    )

    flattened_scores: list[torch.Tensor] = []
    flattened_locations: list[tuple[str, torch.Tensor]] = []
    total_eligible = 0

    for name, parameter in model.named_parameters():
        if name not in eligible:
            continue
        fr = fisher_retain[name].float() / retain_scale
        ff = fisher_forget[name].float() / forget_scale
        score = torch.log(ff + epsilon) - torch.log(fr + epsilon)
        gate = torch.isfinite(score) & torch.isfinite(ff) & (ff >= absolute_threshold)
        scores[name] = score
        gates[name] = gate
        valid_indices = torch.nonzero(gate.reshape(-1), as_tuple=False).squeeze(1)
        if valid_indices.numel() > 0:
            flattened_scores.append(score.reshape(-1)[valid_indices])
            flattened_locations.append((name, valid_indices))
        total_eligible += parameter.numel()

    candidate_count = int(sum(chunk.numel() for chunk in flattened_scores))
    if candidate_count == 0:
        raise RuntimeError("Nessun parametro supera il gate Fisher assoluto.")

    requested_k = max(1, int(round(total_eligible * top_fraction)))
    k = min(requested_k, candidate_count)
    concatenated = torch.cat(flattened_scores)
    selected_global = torch.topk(concatenated, k=k, largest=True, sorted=False).indices
    selected_flags = torch.zeros(candidate_count, dtype=torch.bool)
    selected_flags[selected_global] = True

    masks = {
        name: torch.zeros_like(tensor, dtype=torch.bool)
        for name, tensor in model.state_dict().items()
    }
    cursor = 0
    for name, valid_indices in flattened_locations:
        length = valid_indices.numel()
        chosen = selected_flags[cursor : cursor + length]
        masks[name].view(-1)[valid_indices[chosen]] = True
        cursor += length

    layer_stats = []
    selected_total = 0
    for name, parameter in model.named_parameters():
        selected = int(masks[name].sum().item())
        selected_total += selected
        layer_stats.append(
            {
                "parameter": name,
                "numel": parameter.numel(),
                "selected": selected,
                "selected_fraction": selected / max(parameter.numel(), 1),
                "eligible": name in eligible,
            }
        )

    metadata = {
        "retain_scale_median": retain_scale,
        "forget_scale_median": forget_scale,
        "forget_absolute_quantile": forget_absolute_quantile,
        "forget_absolute_threshold_normalized": absolute_threshold,
        "requested_top_fraction": top_fraction,
        "selected": selected_total,
        "eligible_numel": total_eligible,
        "selected_fraction_of_eligible": selected_total / max(total_eligible, 1),
        "candidate_count_after_gate": candidate_count,
        "layer_stats": layer_stats,
    }
    return masks, scores, metadata


def selective_fisher_dampening(
    model_builder: Callable[[dict[str, torch.Tensor] | None], torch.nn.Module],
    original_state: dict[str, torch.Tensor],
    fisher_retain: dict[str, torch.Tensor],
    fisher_forget: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    *,
    device: torch.device,
    min_factor: float,
    strength: float,
    ratio_power: float,
    epsilon: float = 1e-12,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Applica dampening solo ai pesi selezionati dalla maschera Fisher.

    Ricostruiamo sempre il modello dai pesi originali prima di modificare i
    parametri, cosi' ogni configurazione della ricerca resta indipendente dalle
    altre.
    """
    if not 0 < min_factor <= 1:
        raise ValueError("min_factor deve essere in (0, 1].")
    if not 0 <= strength <= 1:
        raise ValueError("strength deve essere in [0, 1].")
    if ratio_power <= 0:
        raise ValueError("ratio_power deve essere positivo.")

    model = model_builder(original_state).to(device)
    model.eval()

    eligible_names = [name for name, mask in masks.items() if bool(mask.any())]
    retain_values = torch.cat(
        [fisher_retain[name].float().reshape(-1) for name in eligible_names]
    )
    forget_values = torch.cat(
        [fisher_forget[name].float().reshape(-1) for name in eligible_names]
    )
    retain_scale = _positive_median(retain_values)
    forget_scale = _positive_median(forget_values)

    layer_stats: list[dict[str, Any]] = []
    with torch.no_grad():
        named_parameters = dict(model.named_parameters())
        for name, parameter in named_parameters.items():
            mask_cpu = masks.get(name)
            if mask_cpu is None or not bool(mask_cpu.any()):
                continue
            mask = mask_cpu.to(device=device)
            fr = fisher_retain[name].to(device=device, dtype=parameter.dtype) / retain_scale
            ff = fisher_forget[name].to(device=device, dtype=parameter.dtype) / forget_scale
            raw_factor = ((fr + epsilon) / (ff + epsilon)).clamp(max=1.0).pow(ratio_power)
            raw_factor = raw_factor.clamp(min=min_factor, max=1.0)
            factor = 1.0 - strength * (1.0 - raw_factor)
            before = parameter.detach().clone()
            parameter[mask] = parameter[mask] * factor[mask]
            delta = parameter.detach() - before
            layer_stats.append(
                {
                    "parameter": name,
                    "modified": int(mask.sum().item()),
                    "modified_fraction": float(mask.float().mean().item()),
                    "factor_min": float(factor[mask].min().item()),
                    "factor_mean": float(factor[mask].mean().item()),
                    "factor_max": float(factor[mask].max().item()),
                    "delta_l2": float(delta.norm().item()),
                }
            )

    model.eval()
    return model, {
        "min_factor": min_factor,
        "strength": strength,
        "ratio_power": ratio_power,
        "layer_stats": layer_stats,
        "modified_total": int(sum(row["modified"] for row in layer_stats)),
    }


def recalibrate_batchnorm_on_retain(
    model: torch.nn.Module,
    X_retain: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> float:
    """Ricalibra le running statistics BatchNorm usando solo dati retain.

    Non aggiorniamo i pesi: passiamo dati retain attraverso il modello per
    ricostruire running mean e variance coerenti con il modello modificato.
    """
    batchnorm_modules = [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        and module.track_running_stats
    ]
    if not batchnorm_modules:
        model.eval()
        return 0.0
    if len(X_retain) < 2:
        raise ValueError("Servono almeno due esempi per ricalibrare BatchNorm.")

    original_momenta = [module.momentum for module in batchnorm_modules]
    model.eval()
    for module in batchnorm_modules:
        module.reset_running_stats()
        module.momentum = None
        module.train()

    effective_batch = min(max(2, batch_size), len(X_retain))
    drop_last = len(X_retain) % effective_batch == 1
    loader = _make_loader(
        X_retain,
        batch_size=effective_batch,
        shuffle=False,
        seed=0,
        device=device,
        drop_last=drop_last,
    )
    start = time.perf_counter()
    with torch.inference_mode():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            model(xb)
    elapsed = time.perf_counter() - start

    for module, momentum in zip(batchnorm_modules, original_momenta):
        module.momentum = momentum
    model.eval()
    return float(elapsed)


# =============================================================================
# Selective gradient ascent, repair e knowledge distillation
# =============================================================================


def _apply_elementwise_gradient_mask(
    model: torch.nn.Module,
    masks: dict[str, torch.Tensor],
    *,
    update_selected: bool,
) -> None:
    """Applica una maschera elemento-per-elemento ai gradienti.

    Durante il gradient ascent aggiorniamo solo i parametri selezionati; durante
    il repair possiamo invece proteggere quei parametri per non annullare il
    dampening appena applicato.
    """
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        mask = masks.get(name)
        if mask is None:
            if update_selected:
                parameter.grad.zero_()
            continue
        mask_device = mask.to(device=parameter.grad.device)
        if update_selected:
            parameter.grad.mul_(mask_device.to(parameter.grad.dtype))
        else:
            parameter.grad.mul_((~mask_device).to(parameter.grad.dtype))


def selective_gradient_ascent(
    model: torch.nn.Module,
    *,
    X_forget: np.ndarray,
    y_forget: np.ndarray,
    X_retain: np.ndarray,
    teacher_logits_retain: np.ndarray,
    masks: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    learning_rate: float,
    steps: int,
    batch_size: int,
    retain_kd_weight: float,
    gradient_clip: float,
) -> tuple[torch.nn.Module, pd.DataFrame, float]:
    """Esegue gradient ascent controllato sul forget set.

    L'obiettivo aumenta la loss forget, ma include una distillazione retain per
    limitare danni collaterali. Congeliamo BatchNorm in `eval()` per evitare che
    piccoli batch cambino i buffer del modello.
    """
    if steps <= 0:
        return model, pd.DataFrame(), 0.0

    seed_everything(seed)
    model.train()
    # BatchNorm must remain frozen during unlearning updates.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    forget_loader = _make_loader(
        X_forget,
        y_forget,
        batch_size=min(batch_size, len(X_forget)),
        shuffle=True,
        seed=seed,
        device=device,
    )
    retain_loader = _make_loader(
        X_retain,
        teacher_logits_retain,
        batch_size=min(batch_size, len(X_retain)),
        shuffle=True,
        seed=seed + 1,
        device=device,
    )
    forget_iter = iter(forget_loader)
    retain_iter = iter(retain_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for step in range(1, steps + 1):
        try:
            xb_f, yb_f = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            xb_f, yb_f = next(forget_iter)
        try:
            xb_r, teacher_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            xb_r, teacher_r = next(retain_iter)

        xb_f = xb_f.to(device, non_blocking=device.type == "cuda")
        yb_f = yb_f.to(device, non_blocking=device.type == "cuda")
        xb_r = xb_r.to(device, non_blocking=device.type == "cuda")
        teacher_r = teacher_r.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        forget_loss = F.binary_cross_entropy_with_logits(model(xb_f), yb_f)
        retain_kd = F.mse_loss(model(xb_r), teacher_r)
        objective = -forget_loss + retain_kd_weight * retain_kd
        objective.backward()
        _apply_elementwise_gradient_mask(model, masks, update_selected=True)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
        optimizer.step()

        history.append(
            {
                "step": step,
                "forget_bce": float(forget_loss.detach().item()),
                "retain_kd": float(retain_kd.detach().item()),
                "objective": float(objective.detach().item()),
                "gradient_norm": grad_norm,
            }
        )

    elapsed = time.perf_counter() - start
    model.eval()
    return model, pd.DataFrame(history), float(elapsed)


def _parameter_regularization(
    model: torch.nn.Module,
    original_state: dict[str, torch.Tensor],
    dampened_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    *,
    device: torch.device,
    masked_weight: float,
) -> torch.Tensor:
    """Regolarizza i pesi rispetto agli stati originale e dampened.

    I parametri non selezionati restano vicini allo stato originale; quelli
    selezionati possono restare vicini allo stato dampened con un peso dedicato.
    Questo rende il repair meno incline a cancellare l'effetto dell'unlearning.
    """
    total = torch.zeros((), device=device)
    count = 0
    for name, parameter in model.named_parameters():
        original = original_state[name].to(device=device, dtype=parameter.dtype)
        dampened = dampened_state[name].to(device=device, dtype=parameter.dtype)
        mask = masks[name].to(device=device)
        if bool((~mask).any()):
            total = total + (parameter[~mask] - original[~mask]).square().mean()
            count += 1
        if masked_weight > 0 and bool(mask.any()):
            total = total + masked_weight * (
                parameter[mask] - dampened[mask]
            ).square().mean()
            count += 1
    return total / max(count, 1)


def repair_with_distillation(
    model: torch.nn.Module,
    *,
    X_retain: np.ndarray,
    y_retain: np.ndarray,
    teacher_logits_retain: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_forget: np.ndarray,
    y_forget: np.ndarray,
    original_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    attack_proxy: dict[str, Any],
    baseline_p10: float,
    retraining_time: float,
    elapsed_before_repair: float,
    device: torch.device,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight: torch.Tensor,
    lambda_supervised: float,
    lambda_kd: float,
    lambda_parameter: float,
    masked_parameter_weight: float,
    gradient_clip: float,
    freeze_selected: bool,
    utility_floor_ratio: float,
    evaluation_batch_size: int = 2048,
) -> dict[str, Any]:
    """Ripara il modello sul retain set con BCE, distillation e regolarizzazione.

    La BCE mantiene coerenza con le target multilabel, mentre la distillation sui
    logit preserva il comportamento funzionale del modello originale sul retain.
    L'early stopping considera anche il checkpoint iniziale.
    """
    if len(X_retain) != len(teacher_logits_retain):
        raise ValueError("Teacher logits non allineati al retain set.")

    seed_everything(seed)
    dampened_state = state_dict_to_cpu(model)
    loader = _make_loader(
        X_retain,
        y_retain,
        teacher_logits_retain,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    pos_weight = pos_weight.detach().to(device)
    history: list[dict[str, Any]] = []
    repair_start = time.perf_counter()

    initial_metrics = evaluate_candidate(
        model,
        X_validation=X_validation,
        y_validation=y_validation,
        X_forget=X_forget,
        y_forget=y_forget,
        device=device,
        attack_proxy=attack_proxy,
        baseline_p10=baseline_p10,
        retraining_time=retraining_time,
        execution_time=elapsed_before_repair,
        batch_size=evaluation_batch_size,
    )
    initial_metrics.update({"epoch": 0, "train_loss": np.nan})
    history.append(initial_metrics)
    best_state = state_dict_to_cpu(model)
    best_metrics = dict(initial_metrics)
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        running_loss = 0.0
        samples = 0
        for xb, yb, teacher_b in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            teacher_b = teacher_b.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            student_logits = model(xb)
            supervised = F.binary_cross_entropy_with_logits(
                student_logits,
                yb,
                pos_weight=pos_weight,
            )
            kd = F.mse_loss(student_logits, teacher_b)
            parameter_penalty = _parameter_regularization(
                model,
                original_state,
                dampened_state,
                masks,
                device=device,
                masked_weight=masked_parameter_weight,
            )
            loss = (
                lambda_supervised * supervised
                + lambda_kd * kd
                + lambda_parameter * parameter_penalty
            )
            loss.backward()
            if freeze_selected:
                _apply_elementwise_gradient_mask(model, masks, update_selected=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            running_loss += float(loss.detach().item()) * xb.shape[0]
            samples += xb.shape[0]

        cumulative_time = elapsed_before_repair + (time.perf_counter() - repair_start)
        metrics = evaluate_candidate(
            model,
            X_validation=X_validation,
            y_validation=y_validation,
            X_forget=X_forget,
            y_forget=y_forget,
            device=device,
            attack_proxy=attack_proxy,
            baseline_p10=baseline_p10,
            retraining_time=retraining_time,
            execution_time=cumulative_time,
            batch_size=evaluation_batch_size,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(samples, 1),
            }
        )
        history.append(metrics)

        utility_ok = metrics["precision_at_10"] >= baseline_p10 * utility_floor_ratio
        best_utility_ok = (
            best_metrics["precision_at_10"] >= baseline_p10 * utility_floor_ratio
        )
        improves = False
        if utility_ok and not best_utility_ok:
            improves = True
        elif utility_ok == best_utility_ok:
            if metrics["local_score_proxy"] > best_metrics["local_score_proxy"] + 1e-12:
                improves = True
            elif math.isclose(
                metrics["local_score_proxy"],
                best_metrics["local_score_proxy"],
                rel_tol=0,
                abs_tol=1e-12,
            ) and metrics["precision_at_10"] > best_metrics["precision_at_10"]:
                improves = True

        if improves:
            best_state = state_dict_to_cpu(model)
            best_metrics = dict(metrics)
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    repair_elapsed = time.perf_counter() - repair_start
    return {
        "model": model,
        "best_state_dict": best_state,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": pd.DataFrame(history),
        "repair_time_seconds": float(repair_elapsed),
    }


def repair_fixed_epochs(
    model: torch.nn.Module,
    *,
    X_retain: np.ndarray,
    y_retain: np.ndarray,
    teacher_logits_retain: np.ndarray,
    original_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    device: torch.device,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    pos_weight: torch.Tensor,
    lambda_supervised: float,
    lambda_kd: float,
    lambda_parameter: float,
    masked_parameter_weight: float,
    gradient_clip: float,
    freeze_selected: bool,
) -> tuple[torch.nn.Module, pd.DataFrame, float]:
    """Esegue il repair finale per un numero fisso di epoche.

    Usiamo questa variante nella riesecuzione pulita del metodo scelto: non
    guardiamo la validation dentro il timer finale, cosi' la submission non
    include tempo di ricerca o selezione.
    """
    if epochs <= 0:
        model.eval()
        return model, pd.DataFrame(), 0.0
    if len(X_retain) != len(teacher_logits_retain):
        raise ValueError("Teacher logits non allineati al retain set.")

    seed_everything(seed)
    dampened_state = state_dict_to_cpu(model)
    loader = _make_loader(
        X_retain,
        y_retain,
        teacher_logits_retain,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        device=device,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    pos_weight = pos_weight.detach().to(device)
    history: list[dict[str, float]] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

        running_loss = 0.0
        running_supervised = 0.0
        running_kd = 0.0
        samples = 0
        for xb, yb, teacher_b in loader:
            xb = xb.to(device, non_blocking=device.type == "cuda")
            yb = yb.to(device, non_blocking=device.type == "cuda")
            teacher_b = teacher_b.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)
            student_logits = model(xb)
            supervised = F.binary_cross_entropy_with_logits(
                student_logits, yb, pos_weight=pos_weight
            )
            kd = F.mse_loss(student_logits, teacher_b)
            parameter_penalty = _parameter_regularization(
                model,
                original_state,
                dampened_state,
                masks,
                device=device,
                masked_weight=masked_parameter_weight,
            )
            loss = (
                lambda_supervised * supervised
                + lambda_kd * kd
                + lambda_parameter * parameter_penalty
            )
            loss.backward()
            if freeze_selected:
                _apply_elementwise_gradient_mask(model, masks, update_selected=False)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_n = xb.shape[0]
            running_loss += float(loss.detach().item()) * batch_n
            running_supervised += float(supervised.detach().item()) * batch_n
            running_kd += float(kd.detach().item()) * batch_n
            samples += batch_n

        history.append(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(samples, 1),
                "supervised_bce": running_supervised / max(samples, 1),
                "distillation_mse": running_kd / max(samples, 1),
            }
        )

    elapsed = time.perf_counter() - start
    model.eval()
    return model, pd.DataFrame(history), float(elapsed)


def execute_final_unlearning_config(
    config: dict[str, Any],
    *,
    fixed_repair_epochs: int,
    model_builder: Callable[[dict[str, torch.Tensor] | None], torch.nn.Module],
    original_state: dict[str, torch.Tensor],
    fisher_retain: dict[str, torch.Tensor],
    fisher_forget: dict[str, torch.Tensor],
    X_retain: np.ndarray,
    y_retain: np.ndarray,
    teacher_logits_retain: np.ndarray,
    X_forget: np.ndarray,
    y_forget: np.ndarray,
    pos_weight: torch.Tensor,
    device: torch.device,
    seed: int,
    precompute_time_seconds: float,
) -> dict[str, Any]:
    """Riesegue una configurazione scelta misurando solo il metodo finale.

    La funzione parte dai pesi originali, applica dampening, eventuale gradient
    ascent, ricalibrazione BatchNorm e repair a epoche fisse. Non usa la
    validation per scegliere checkpoint.
    """
    seed_everything(seed)
    algorithm_start = time.perf_counter()

    original_model_for_mask = model_builder(original_state).to(device).eval()
    masks, _, mask_metadata = build_fisher_mask(
        original_model_for_mask,
        fisher_retain,
        fisher_forget,
        top_fraction=float(config["top_fraction"]),
        forget_absolute_quantile=float(config.get("forget_absolute_quantile", 0.50)),
        include_bias=bool(config.get("include_bias", False)),
        include_batchnorm_affine=bool(config.get("include_batchnorm_affine", False)),
    )
    del original_model_for_mask

    model, dampening_metadata = selective_fisher_dampening(
        model_builder,
        original_state,
        fisher_retain,
        fisher_forget,
        masks,
        device=device,
        min_factor=float(config["min_factor"]),
        strength=float(config.get("dampening_strength", 1.0)),
        ratio_power=float(config.get("ratio_power", 1.0)),
    )

    ga_history = pd.DataFrame()
    ga_time = 0.0
    if int(config.get("ga_steps", 0)) > 0:
        model, ga_history, ga_time = selective_gradient_ascent(
            model,
            X_forget=X_forget,
            y_forget=y_forget,
            X_retain=X_retain,
            teacher_logits_retain=teacher_logits_retain,
            masks=masks,
            device=device,
            seed=seed,
            learning_rate=float(config["ga_learning_rate"]),
            steps=int(config["ga_steps"]),
            batch_size=int(config.get("ga_batch_size", 128)),
            retain_kd_weight=float(config.get("ga_retain_kd_weight", 1.0)),
            gradient_clip=float(config.get("gradient_clip", 1.0)),
        )

    model, repair_history, repair_time = repair_fixed_epochs(
        model,
        X_retain=X_retain,
        y_retain=y_retain,
        teacher_logits_retain=teacher_logits_retain,
        original_state=original_state,
        masks=masks,
        device=device,
        seed=seed,
        learning_rate=float(config["repair_learning_rate"]),
        weight_decay=float(config.get("repair_weight_decay", 0.0)),
        batch_size=int(config.get("repair_batch_size", 512)),
        epochs=int(fixed_repair_epochs),
        pos_weight=pos_weight,
        lambda_supervised=float(config.get("lambda_supervised", 1.0)),
        lambda_kd=float(config.get("lambda_kd", 0.5)),
        lambda_parameter=float(config.get("lambda_parameter", 1e-4)),
        masked_parameter_weight=float(config.get("masked_parameter_weight", 1.0)),
        gradient_clip=float(config.get("gradient_clip", 1.0)),
        freeze_selected=bool(config.get("freeze_selected", True)),
    )

    bn_recalibration_time = 0.0
    if bool(config.get("recalibrate_batchnorm", True)):
        bn_recalibration_time = recalibrate_batchnorm_on_retain(
            model,
            X_retain,
            device=device,
            batch_size=int(config.get("bn_recalibration_batch_size", 2048)),
        )

    # algorithm_start includes mask/dampening, optional GA, fixed repair and
    # intentional BatchNorm recalibration, but excludes post-hoc evaluation.
    post_precompute_time = time.perf_counter() - algorithm_start
    total_time = float(precompute_time_seconds + post_precompute_time)
    return {
        "model": model,
        "state_dict": state_dict_to_cpu(model),
        "execution_time_seconds": total_time,
        "fixed_repair_epochs": int(fixed_repair_epochs),
        "mask_metadata": mask_metadata,
        "dampening_metadata": dampening_metadata,
        "repair_history": repair_history,
        "ga_history": ga_history,
        "ga_time_seconds": float(ga_time),
        "repair_time_seconds": float(repair_time),
        "bn_recalibration_time_seconds": float(bn_recalibration_time),
        "masks": {name: mask.cpu() for name, mask in masks.items()},
    }


# =============================================================================
# Esecuzione candidati e ricerca progressiva
# =============================================================================


def precompute_teacher_logits(
    teacher_model: torch.nn.Module,
    X_retain: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    """Calcola una volta i logit teacher sul retain set.

    Li riusiamo durante il repair per knowledge distillation. L'ordine deve
    restare identico a `X_retain`, quindi qui non applichiamo shuffle.
    """
    start = time.perf_counter()
    logits = predict_logits_numpy(
        teacher_model,
        X_retain,
        device=device,
        batch_size=batch_size,
    )
    return logits.astype(np.float32, copy=False), float(time.perf_counter() - start)


def execute_unlearning_config(
    config: dict[str, Any],
    *,
    model_builder: Callable[[dict[str, torch.Tensor] | None], torch.nn.Module],
    original_state: dict[str, torch.Tensor],
    fisher_retain: dict[str, torch.Tensor],
    fisher_forget: dict[str, torch.Tensor],
    X_retain: np.ndarray,
    y_retain: np.ndarray,
    teacher_logits_retain: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    X_forget: np.ndarray,
    y_forget: np.ndarray,
    attack_proxy: dict[str, Any],
    baseline_p10: float,
    retraining_time: float,
    pos_weight: torch.Tensor,
    device: torch.device,
    seed: int,
    fisher_time_seconds: float = 0.0,
) -> dict[str, Any]:
    """Esegue una configurazione candidata durante la ricerca.

    Ogni candidato riparte dai pesi originali, applica la maschera Fisher,
    valuta eventuale gradient ascent e fa repair con early stopping su validation.
    La ricerca non rappresenta ancora il tempo dichiarato della submission.
    """
    seed_everything(seed)
    start = time.perf_counter()
    original_model_for_mask = model_builder(original_state).to(device).eval()
    masks, _, mask_metadata = build_fisher_mask(
        original_model_for_mask,
        fisher_retain,
        fisher_forget,
        top_fraction=float(config["top_fraction"]),
        forget_absolute_quantile=float(config.get("forget_absolute_quantile", 0.50)),
        include_bias=bool(config.get("include_bias", False)),
        include_batchnorm_affine=bool(config.get("include_batchnorm_affine", False)),
    )
    del original_model_for_mask

    model, dampening_metadata = selective_fisher_dampening(
        model_builder,
        original_state,
        fisher_retain,
        fisher_forget,
        masks,
        device=device,
        min_factor=float(config["min_factor"]),
        strength=float(config.get("dampening_strength", 1.0)),
        ratio_power=float(config.get("ratio_power", 1.0)),
    )
    elapsed = fisher_time_seconds + (time.perf_counter() - start)
    ga_history = pd.DataFrame()

    if int(config.get("ga_steps", 0)) > 0:
        model, ga_history, ga_time = selective_gradient_ascent(
            model,
            X_forget=X_forget,
            y_forget=y_forget,
            X_retain=X_retain,
            teacher_logits_retain=teacher_logits_retain,
            masks=masks,
            device=device,
            seed=seed,
            learning_rate=float(config["ga_learning_rate"]),
            steps=int(config["ga_steps"]),
            batch_size=int(config.get("ga_batch_size", 128)),
            retain_kd_weight=float(config.get("ga_retain_kd_weight", 1.0)),
            gradient_clip=float(config.get("gradient_clip", 1.0)),
        )
        elapsed += ga_time

    repair_epochs = int(config.get("repair_epochs", 0))
    if repair_epochs > 0:
        repair_result = repair_with_distillation(
            model,
            X_retain=X_retain,
            y_retain=y_retain,
            teacher_logits_retain=teacher_logits_retain,
            X_validation=X_validation,
            y_validation=y_validation,
            X_forget=X_forget,
            y_forget=y_forget,
            original_state=original_state,
            masks=masks,
            attack_proxy=attack_proxy,
            baseline_p10=baseline_p10,
            retraining_time=retraining_time,
            elapsed_before_repair=elapsed,
            device=device,
            seed=seed,
            learning_rate=float(config["repair_learning_rate"]),
            weight_decay=float(config.get("repair_weight_decay", 0.0)),
            batch_size=int(config.get("repair_batch_size", 512)),
            max_epochs=repair_epochs,
            patience=int(config.get("repair_patience", 2)),
            pos_weight=pos_weight,
            lambda_supervised=float(config.get("lambda_supervised", 1.0)),
            lambda_kd=float(config.get("lambda_kd", 0.5)),
            lambda_parameter=float(config.get("lambda_parameter", 1e-4)),
            masked_parameter_weight=float(config.get("masked_parameter_weight", 1.0)),
            gradient_clip=float(config.get("gradient_clip", 1.0)),
            freeze_selected=bool(config.get("freeze_selected", True)),
            utility_floor_ratio=float(config.get("utility_floor_ratio", 0.985)),
        )
        model = repair_result["model"]
        elapsed += repair_result["repair_time_seconds"]
        metrics = dict(repair_result["best_metrics"])
        repair_history = repair_result["history"]
        best_epoch = repair_result["best_epoch"]
    else:
        metrics = evaluate_candidate(
            model,
            X_validation=X_validation,
            y_validation=y_validation,
            X_forget=X_forget,
            y_forget=y_forget,
            device=device,
            attack_proxy=attack_proxy,
            baseline_p10=baseline_p10,
            retraining_time=retraining_time,
            execution_time=elapsed,
        )
        repair_history = pd.DataFrame()
        best_epoch = 0

    bn_recalibration_time = 0.0
    if bool(config.get("recalibrate_batchnorm", True)):
        bn_recalibration_time = recalibrate_batchnorm_on_retain(
            model,
            X_retain,
            device=device,
            batch_size=int(config.get("bn_recalibration_batch_size", 2048)),
        )
        elapsed += bn_recalibration_time
        metrics = evaluate_candidate(
            model,
            X_validation=X_validation,
            y_validation=y_validation,
            X_forget=X_forget,
            y_forget=y_forget,
            device=device,
            attack_proxy=attack_proxy,
            baseline_p10=baseline_p10,
            retraining_time=retraining_time,
            execution_time=elapsed,
        )

    metrics["execution_time_seconds"] = float(elapsed)
    metrics["best_epoch"] = int(best_epoch)
    metrics["bn_recalibration_time_seconds"] = float(bn_recalibration_time)
    metrics["utility_floor_pass"] = bool(
        metrics["precision_at_10"]
        >= baseline_p10 * float(config.get("utility_floor_ratio", 0.985))
    )

    return {
        "model": model,
        "state_dict": state_dict_to_cpu(model),
        "metrics": metrics,
        "config": deepcopy(config),
        "mask_metadata": mask_metadata,
        "dampening_metadata": dampening_metadata,
        "repair_history": repair_history,
        "ga_history": ga_history,
        "masks": {k: v.cpu() for k, v in masks.items()},
    }


def default_search_configs(
    *,
    base_learning_rate: float,
    base_weight_decay: float,
    train_batch_size: int,
) -> list[dict[str, Any]]:
    """Costruisce una griglia compatta di configurazioni candidate.

    Variamo frazione selezionata, dampening minimo e ricalibrazione BatchNorm.
    La griglia resta volutamente piccola per mantenere la ricerca presentabile e
    sostenibile anche su CPU.
    """
    repair_lr = max(base_learning_rate * 0.05, 1e-6)
    common = {
        "forget_absolute_quantile": 0.50,
        "dampening_strength": 1.0,
        "ratio_power": 1.0,
        "repair_learning_rate": repair_lr,
        "repair_weight_decay": min(base_weight_decay, 1e-4),
        "repair_batch_size": min(train_batch_size, 1024),
        "repair_epochs": 6,
        "repair_patience": 2,
        "lambda_supervised": 1.0,
        "lambda_kd": 0.50,
        "lambda_parameter": 1e-4,
        "masked_parameter_weight": 1.0,
        "gradient_clip": 1.0,
        "freeze_selected": True,
        "utility_floor_ratio": 0.985,
        "ga_steps": 0,
        "recalibrate_batchnorm": True,
        "bn_recalibration_batch_size": 2048,
    }
    configs = []
    for top_fraction, min_factor in [
        (0.0050, 0.90),
        (0.0100, 0.90),
        (0.0100, 0.82),
        (0.0200, 0.82),
    ]:
        for recalibrate_bn in (False, True):
            config = dict(common)
            config.update(
                {
                    "name": (
                        f"ssd_repair_tf{top_fraction:g}_mf{min_factor:g}"
                        f"_bn{int(recalibrate_bn)}"
                    ),
                    "top_fraction": top_fraction,
                    "min_factor": min_factor,
                    "recalibrate_batchnorm": recalibrate_bn,
                }
            )
            configs.append(config)
    return configs


def select_best_result(
    results: list[dict[str, Any]],
    *,
    baseline_p10: float,
    utility_floor_ratio: float,
) -> dict[str, Any]:
    """Seleziona il miglior risultato dando priorita' al vincolo di utility."""
    if not results:
        raise ValueError("Nessun risultato da selezionare.")

    feasible = [
        result
        for result in results
        if result["metrics"]["precision_at_10"] >= baseline_p10 * utility_floor_ratio
    ]
    pool = feasible if feasible else results
    return max(
        pool,
        key=lambda result: (
            result["metrics"]["local_score_proxy"],
            result["metrics"]["proxy_privacy"],
            result["metrics"]["precision_at_10"],
            -result["metrics"]["execution_time_seconds"],
        ),
    )


def progressive_search(
    configs: Iterable[dict[str, Any]],
    *,
    execute_kwargs: dict[str, Any],
    baseline_p10: float,
    utility_floor_ratio: float = 0.985,
    add_gradient_ascent_variants: int = 2,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    """Esegue la ricerca progressiva e poche varianti gradient-ascent.

    Stampiamo una riga compatta per configurazione. Il runner riesegue poi il
    candidato scelto da zero, quindi il tempo della ricerca non finisce nella
    submission.
    """
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for index, config in enumerate(configs, start=1):
        result = execute_unlearning_config(config, **execute_kwargs)
        results.append(result)
        rows.append({"config_index": index, **config, **result["metrics"]})
        print(
            f"[{index}] {config['name']} | "
            f"P@10={result['metrics']['precision_at_10']:.6f} | "
            f"privacy={result['metrics']['proxy_privacy']:.6f} | "
            f"score={result['metrics']['local_score_proxy']:.6f}"
        )
        result.pop("model", None)
        clear_memory()

    ranked = sorted(
        results,
        key=lambda r: r["metrics"]["local_score_proxy"],
        reverse=True,
    )[:add_gradient_ascent_variants]

    for base in ranked:
        ga_config = deepcopy(base["config"])
        ga_config.update(
            {
                "name": base["config"]["name"] + "_ga",
                "ga_steps": 4,
                "ga_learning_rate": max(
                    float(base["config"]["repair_learning_rate"]) * 0.10,
                    1e-7,
                ),
                "ga_batch_size": 128,
                "ga_retain_kd_weight": 1.0,
            }
        )
        result = execute_unlearning_config(ga_config, **execute_kwargs)
        results.append(result)
        rows.append(
            {
                "config_index": len(rows) + 1,
                **ga_config,
                **result["metrics"],
            }
        )
        print(
            f"[{len(rows)}] {ga_config['name']} | "
            f"P@10={result['metrics']['precision_at_10']:.6f} | "
            f"privacy={result['metrics']['proxy_privacy']:.6f} | "
            f"score={result['metrics']['local_score_proxy']:.6f}"
        )
        result.pop("model", None)
        clear_memory()

    best = select_best_result(
        results,
        baseline_p10=baseline_p10,
        utility_floor_ratio=utility_floor_ratio,
    )
    comparison = pd.DataFrame(rows).sort_values(
        ["utility_floor_pass", "local_score_proxy", "proxy_privacy", "precision_at_10"],
        ascending=[False, False, False, False],
    )
    return best, comparison, results


# =============================================================================
# Validazione artifact e creazione della submission
# =============================================================================


def validate_model_artifact(
    artifact_path: Path,
    *,
    model_class: type[torch.nn.Module],
) -> dict[str, Any]:
    """Verifica che un artifact sia completo e caricabile con `strict=True`."""
    with open(artifact_path, "rb") as handle:
        payload = pickle.load(handle)
    required = {"state_dict", "architecture", "best_hyperparameters", "model_class_source"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Artifact incompleto: {sorted(missing)}")
    architecture = payload["architecture"]
    model = model_class(
        architecture["input_dim"],
        architecture["hidden_layers"],
        architecture["num_outputs"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return payload


def save_submission(
    *,
    submission_dir: str | Path,
    final_state_dict: dict[str, torch.Tensor],
    execution_time_seconds: float,
    validation_ids: pd.Series | np.ndarray | list[Any],
    id_column: str,
    original_payload: dict[str, Any],
    selected_config: dict[str, Any],
    final_metrics: dict[str, Any],
    fisher_metadata: dict[str, Any],
    model_class: type[torch.nn.Module],
) -> dict[str, Path]:
    """Crea e valida i tre file richiesti per la submission.

    I tensori vengono salvati su CPU, il tempo viene arrotondato per eccesso e
    `validation_ids.csv` viene riletto subito per controllare header e formato.
    """
    submission_dir = Path(submission_dir)
    submission_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = submission_dir / "model_artifact"
    execution_path = submission_dir / "execution_time.txt"
    validation_path = submission_dir / "validation_ids.csv"

    payload = {
        "state_dict": clone_state_dict(final_state_dict),
        "architecture": deepcopy(original_payload["architecture"]),
        "best_hyperparameters": {
            **deepcopy(original_payload["best_hyperparameters"]),
            "unlearning_method": str(selected_config.get("name", "unlearning_candidate")),
            "selected_config": deepcopy(selected_config),
            "local_metrics": {
                key: float(value) if isinstance(value, (np.floating, float)) else value
                for key, value in final_metrics.items()
                if isinstance(value, (int, float, bool, np.integer, np.floating))
            },
            "execution_time_seconds": float(execution_time_seconds),
        },
        "model_class_source": original_payload["model_class_source"],
        "unlearning_metadata": {
            "selected_config": deepcopy(selected_config),
            "final_metrics": deepcopy(final_metrics),
            "fisher_metadata": deepcopy(fisher_metadata),
            "official_mia_replication": False,
            "privacy_metric_type": "retrained-reference proxy",
        },
    }

    with open(artifact_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    execution_path.write_text(str(int(math.ceil(execution_time_seconds))), encoding="utf-8")
    pd.DataFrame({id_column: np.asarray(validation_ids)}).to_csv(
        validation_path, index=False
    )

    validate_model_artifact(artifact_path, model_class=model_class)
    if execution_path.read_text(encoding="utf-8").strip() != str(
        int(math.ceil(execution_time_seconds))
    ):
        raise RuntimeError("execution_time.txt non valido.")
    loaded_validation = pd.read_csv(validation_path)
    if list(loaded_validation.columns) != [id_column]:
        raise RuntimeError("validation_ids.csv non valido.")

    return {
        "model_artifact": artifact_path,
        "execution_time": execution_path,
        "validation_ids": validation_path,
    }


def write_json(path: str | Path, payload: Any) -> None:
    """Scrive JSON leggibile gestendo tipi NumPy, PyTorch e `Path`."""
    def default(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        raise TypeError(f"Tipo non serializzabile: {type(value)!r}")

    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )
