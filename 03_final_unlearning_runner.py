"""
Runner finale da eseguire dopo le baseline del notebook.

Uso previsto nel notebook:

    %run -i final_unlearning_runner.py

Il runner usa le variabili gia' create da `prova.ipynb`, esegue la ricerca delle
configurazioni di unlearning e genera la submission finale nella cartella:

    outputs/unlearning_final/submission/
"""

import math
from pathlib import Path

import pandas as pd
from IPython.display import display

from final_unlearning_pipeline import (
    clear_memory,
    default_search_configs,
    empirical_fisher_diagonal,
    evaluate_candidate,
    execute_final_unlearning_config,
    fit_reference_attack_proxy,
    precompute_teacher_logits,
    progressive_search,
    save_submission,
    seed_everything,
    state_dict_to_cpu,
    write_json,
)


# =============================================================================
# 1. Verifica delle variabili ricevute dal notebook
# =============================================================================

# Verifichiamo subito che il notebook abbia gia' definito tutte le variabili
# necessarie. In questo modo evitiamo errori poco chiari a meta' della ricerca.
REQUIRED_NOTEBOOK_GLOBALS = [
    "SEED",
    "ID_COL",
    "output_dir",
    "device",
    "model",
    "model_payload",
    "architecture",
    "build_model",
    "X_retain",
    "y_retain",
    "X_validation",
    "y_validation",
    "X_forget",
    "y_forget",
    "validation_df",
    "baseline_p10",
    "baseline_forget",
    "retrained_model",
    "retrained_validation",
    "retrained_forget",
    "retraining_time",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "TRAIN_BATCH_SIZE",
    "EVAL_BATCH_SIZE",
    "pos_weight",
    "DynamicMLP",
]

missing_globals = [name for name in REQUIRED_NOTEBOOK_GLOBALS if name not in globals()]
if missing_globals:
    raise RuntimeError(
        "Esegui prima tutte le celle del notebook baseline. Variabili mancanti: "
        + ", ".join(missing_globals)
    )


# =============================================================================
# 2. Configurazione iniziale e stato originale
# =============================================================================

seed_everything(SEED)

UNLEARNING_DIR = Path(output_dir) / "unlearning_final"
SUBMISSION_DIR = UNLEARNING_DIR / "submission"
UNLEARNING_DIR.mkdir(parents=True, exist_ok=True)

# Ogni esperimento deve ripartire dallo stesso modello originale. Conserviamo lo
# stato su CPU per evitare modifiche accidentali e pressione inutile sulla GPU.
ORIGINAL_STATE_CPU = state_dict_to_cpu(model)

print("[1/8] Contesto verificato")
print(f"Output unlearning: {UNLEARNING_DIR}")


# =============================================================================
# 3. Proxy privacy basata sul modello retrained
# =============================================================================

# La validation non e' un vero non-member set garantito. Per questo costruiamo
# una proxy locale sul forget set, confrontando modello originale e retrained.
original_forget_logits = baseline_forget["logits"]
retrained_forget_logits = retrained_forget["logits"]

attack_proxy = fit_reference_attack_proxy(
    original_forget_logits,
    retrained_forget_logits,
    y_forget,
    seed=SEED,
)
print(f"[2/8] Proxy privacy pronta | grouped-CV AUC={attack_proxy['cv_auc']:.6f}")


# =============================================================================
# 4. Quantita' condivise: teacher logits e Fisher diagonali
# =============================================================================

TEACHER_BATCH_SIZE = max(1024, min(4096, EVAL_BATCH_SIZE))
FISHER_BATCH_SIZE = 32
FISHER_RETAIN_SAMPLE = min(len(X_retain), 4096)
FISHER_FORGET_SAMPLE = len(X_forget)
UTILITY_FLOOR_RATIO = 0.985

print("[3/8] Precalcolo teacher logits sul retain set")
teacher_logits_retain, teacher_time_search = precompute_teacher_logits(
    model,
    X_retain,
    device=device,
    batch_size=TEACHER_BATCH_SIZE,
)

print("[4/8] Calcolo Fisher retain e forget")
fisher_retain, fisher_retain_meta = empirical_fisher_diagonal(
    model,
    X_retain,
    y_retain,
    device=device,
    sample_size=FISHER_RETAIN_SAMPLE,
    batch_size=FISHER_BATCH_SIZE,
    seed=SEED,
    pos_weight=None,
    include_bias=False,
    include_batchnorm_affine=False,
)
fisher_forget, fisher_forget_meta = empirical_fisher_diagonal(
    model,
    X_forget,
    y_forget,
    device=device,
    sample_size=FISHER_FORGET_SAMPLE,
    batch_size=FISHER_BATCH_SIZE,
    seed=SEED + 1,
    pos_weight=None,
    include_bias=False,
    include_batchnorm_affine=False,
)

shared_method_time = (
    teacher_time_search
    + fisher_retain_meta["elapsed_seconds"]
    + fisher_forget_meta["elapsed_seconds"]
)


# =============================================================================
# 5. Definizione delle configurazioni sperimentali
# =============================================================================

# La griglia e' volutamente compatta: proviamo dampening piu' o meno aggressivo,
# con o senza ricalibrazione BatchNorm. Le varianti gradient-ascent vengono
# aggiunte dopo solo sui candidati migliori.
search_configs = default_search_configs(
    base_learning_rate=LEARNING_RATE,
    base_weight_decay=WEIGHT_DECAY,
    train_batch_size=TRAIN_BATCH_SIZE,
)

execute_kwargs = {
    "model_builder": build_model,
    "original_state": ORIGINAL_STATE_CPU,
    "fisher_retain": fisher_retain,
    "fisher_forget": fisher_forget,
    "X_retain": X_retain,
    "y_retain": y_retain,
    "teacher_logits_retain": teacher_logits_retain,
    "X_validation": X_validation,
    "y_validation": y_validation,
    "X_forget": X_forget,
    "y_forget": y_forget,
    "attack_proxy": attack_proxy,
    "baseline_p10": baseline_p10,
    "retraining_time": retraining_time,
    "pos_weight": pos_weight,
    "device": device,
    "seed": SEED,
    "fisher_time_seconds": shared_method_time,
}


# =============================================================================
# 6. Ricerca progressiva e configurazioni dominate
# =============================================================================

# La ricerca serve soltanto a scegliere una configurazione. Il metodo finale verra'
# rieseguito dopo da zero, senza includere nel tempo dichiarato la diagnostica.
print("[5/8] Ricerca configurazioni SSD/repair")
best_hybrid, search_comparison, search_results = progressive_search(
    search_configs,
    execute_kwargs=execute_kwargs,
    baseline_p10=baseline_p10,
    utility_floor_ratio=UTILITY_FLOOR_RATIO,
    add_gradient_ascent_variants=2,
)
search_comparison.to_csv(UNLEARNING_DIR / "search_comparison.csv", index=False)

dominated_count = int((~search_comparison["utility_floor_pass"]).sum())
print(f"Configurazioni sotto utility floor: {dominated_count}/{len(search_comparison)}")


# =============================================================================
# 7. Valutazione finaliste e selezione del metodo finale
# =============================================================================

# Il retraining da zero e' il riferimento piu' pulito per il forget set. Lo
# confrontiamo sempre con il miglior ibrido invece di imporre SSD a priori.
retrained_metrics_for_selection = evaluate_candidate(
    retrained_model,
    X_validation=X_validation,
    y_validation=y_validation,
    X_forget=X_forget,
    y_forget=y_forget,
    device=device,
    attack_proxy=attack_proxy,
    baseline_p10=baseline_p10,
    retraining_time=retraining_time,
    execution_time=retraining_time,
)
retrained_metrics_for_selection["execution_time_seconds"] = float(retraining_time)
retrained_metrics_for_selection["best_epoch"] = int(
    retraining_result["best_epoch"] if "retraining_result" in globals() else best_epoch
)
retrained_metrics_for_selection["utility_floor_pass"] = bool(
    retrained_metrics_for_selection["precision_at_10"]
    >= baseline_p10 * UTILITY_FLOOR_RATIO
)

hybrid_metrics = best_hybrid["metrics"]
hybrid_feasible = bool(hybrid_metrics["utility_floor_pass"])
retrained_feasible = bool(retrained_metrics_for_selection["utility_floor_pass"])

if retrained_feasible and not hybrid_feasible:
    selected_kind = "retraining_from_scratch"
elif hybrid_feasible and not retrained_feasible:
    selected_kind = "hybrid"
elif retrained_metrics_for_selection["local_score_proxy"] > hybrid_metrics["local_score_proxy"]:
    selected_kind = "retraining_from_scratch"
else:
    selected_kind = "hybrid"

final_comparison = pd.DataFrame(
    [
        {"candidate": "best_hybrid", "config": best_hybrid["config"]["name"], **hybrid_metrics},
        {
            "candidate": "retraining_from_scratch",
            "config": "retraining_from_scratch",
            **retrained_metrics_for_selection,
        },
    ]
).sort_values(
    ["utility_floor_pass", "local_score_proxy", "proxy_privacy", "precision_at_10"],
    ascending=[False, False, False, False],
)
final_comparison.to_csv(UNLEARNING_DIR / "final_candidate_comparison.csv", index=False)

print("[6/8] Finaliste valutate")
display(final_comparison)
print("Metodo selezionato:", selected_kind)


# =============================================================================
# 8. Riesecuzione pulita e misurazione del tempo finale
# =============================================================================

if selected_kind == "retraining_from_scratch":
    selected_config = {
        "name": "retraining_from_scratch",
        "best_epoch": int(retrained_metrics_for_selection["best_epoch"]),
        "utility_floor_ratio": UTILITY_FLOOR_RATIO,
    }
    final_state = state_dict_to_cpu(retrained_model)
    final_metrics = retrained_metrics_for_selection
    final_execution_time = float(retraining_time)
    final_fisher_metadata = {
        "used": False,
        "reason": "retraining_from_scratch selected by the local weighted proxy",
    }
    final_history = (
        retraining_result["history"]
        if "retraining_result" in globals()
        else pd.DataFrame()
    )
    final_ga_history = pd.DataFrame()
else:
    selected_config = dict(best_hybrid["config"])

    # Misuriamo solo la procedura finale: ricalcoliamo teacher logits e Fisher,
    # poi eseguiamo il repair a epoche fisse scelte dalla ricerca.
    print("[7/8] Riesecuzione pulita del metodo ibrido selezionato")
    teacher_logits_final, teacher_time_final = precompute_teacher_logits(
        model,
        X_retain,
        device=device,
        batch_size=TEACHER_BATCH_SIZE,
    )
    fisher_retain_final, fisher_retain_meta_final = empirical_fisher_diagonal(
        model,
        X_retain,
        y_retain,
        device=device,
        sample_size=FISHER_RETAIN_SAMPLE,
        batch_size=FISHER_BATCH_SIZE,
        seed=SEED,
        pos_weight=None,
        include_bias=False,
        include_batchnorm_affine=False,
    )
    fisher_forget_final, fisher_forget_meta_final = empirical_fisher_diagonal(
        model,
        X_forget,
        y_forget,
        device=device,
        sample_size=FISHER_FORGET_SAMPLE,
        batch_size=FISHER_BATCH_SIZE,
        seed=SEED + 1,
        pos_weight=None,
        include_bias=False,
        include_batchnorm_affine=False,
    )
    fixed_time = (
        teacher_time_final
        + fisher_retain_meta_final["elapsed_seconds"]
        + fisher_forget_meta_final["elapsed_seconds"]
    )

    fixed_repair_epochs = int(best_hybrid["metrics"].get("best_epoch", 0))
    selected_config["fixed_repair_epochs"] = fixed_repair_epochs
    final_result = execute_final_unlearning_config(
        selected_config,
        fixed_repair_epochs=fixed_repair_epochs,
        model_builder=build_model,
        original_state=ORIGINAL_STATE_CPU,
        fisher_retain=fisher_retain_final,
        fisher_forget=fisher_forget_final,
        X_retain=X_retain,
        y_retain=y_retain,
        teacher_logits_retain=teacher_logits_final,
        X_forget=X_forget,
        y_forget=y_forget,
        pos_weight=pos_weight,
        device=device,
        seed=SEED,
        precompute_time_seconds=fixed_time,
    )
    final_state = final_result["state_dict"]
    final_execution_time = float(final_result["execution_time_seconds"])
    final_model_for_evaluation = final_result["model"]
    final_metrics = evaluate_candidate(
        final_model_for_evaluation,
        X_validation=X_validation,
        y_validation=y_validation,
        X_forget=X_forget,
        y_forget=y_forget,
        device=device,
        attack_proxy=attack_proxy,
        baseline_p10=baseline_p10,
        retraining_time=retraining_time,
        execution_time=final_execution_time,
    )
    final_metrics["execution_time_seconds"] = final_execution_time
    final_metrics["best_epoch"] = fixed_repair_epochs
    final_metrics["utility_floor_pass"] = bool(
        final_metrics["precision_at_10"] >= baseline_p10 * UTILITY_FLOOR_RATIO
    )
    final_fisher_metadata = {
        "used": True,
        "teacher_time_seconds": teacher_time_final,
        "retain": fisher_retain_meta_final,
        "forget": fisher_forget_meta_final,
        "mask": final_result["mask_metadata"],
        "dampening": final_result["dampening_metadata"],
    }
    final_history = final_result["repair_history"]
    final_ga_history = final_result["ga_history"]


# =============================================================================
# 9. Creazione e verifica della submission
# =============================================================================

print("[8/8] Creazione submission")
submission_paths = save_submission(
    submission_dir=SUBMISSION_DIR,
    final_state_dict=final_state,
    execution_time_seconds=final_execution_time,
    validation_ids=validation_df[ID_COL].to_numpy(),
    id_column=ID_COL,
    original_payload=model_payload,
    selected_config=selected_config,
    final_metrics=final_metrics,
    fisher_metadata=final_fisher_metadata,
    model_class=DynamicMLP,
)

if not final_history.empty:
    final_history.to_csv(UNLEARNING_DIR / "final_history.csv", index=False)
if not final_ga_history.empty:
    final_ga_history.to_csv(UNLEARNING_DIR / "final_ga_history.csv", index=False)

write_json(UNLEARNING_DIR / "selected_config.json", selected_config)
write_json(UNLEARNING_DIR / "final_metrics.json", final_metrics)
write_json(UNLEARNING_DIR / "fisher_metadata.json", final_fisher_metadata)

print("\nSUBMISSION PRONTA")
for key, path in submission_paths.items():
    print(f"{key}: {path}")
print(f"Tempo dichiarato: {int(math.ceil(final_execution_time))} s (file arrotondato per eccesso)")
print("Metriche locali finali:")
display(pd.DataFrame([final_metrics]))

# I modelli candidati della ricerca non servono piu'. Li rimuoviamo dai risultati
# conservati e liberiamo memoria prima di tornare al notebook.
for candidate in search_results:
    candidate.pop("model", None)
clear_memory()
