"""Select a strict multi-seed hybrid recommendation without canonical promotion."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.hybrid_recommendation import (  # noqa: E402
    assert_noncanonical_config_path,
    recommend_hybrid_configuration,
)
from machine_unlearning.workflow import write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit, reviewable recommendation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Seleziona una configurazione ibrida multi-seed con vincoli rigidi; "
            "non modifica mai configs/final_config.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/stage4_multiseed/all_candidates_summary.csv"),
        help="Riepilogo prodotto da summarize_all_candidates.py.",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("outputs/stage4_multiseed/all_candidates_all_seeds.csv"),
        help="Risultati grezzi aggregati, necessari per seed ed epoche.",
    )
    parser.add_argument(
        "--search-config",
        type=Path,
        default=Path("configs/search_stage4_multiseed.json"),
        help="Configurazione di ricerca sorgente per Fisher e valori condivisi.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=[92, 93, 94, 95, 96],
        help="Set esatto di seed richiesto per l'eleggibilita'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=92,
        help="Seed incorporato nella configurazione finale riproducibile.",
    )
    parser.add_argument(
        "--recommendation-output",
        type=Path,
        default=Path("outputs/stage4_multiseed/hybrid_recommendation.json"),
        help="JSON di raccomandazione e audit.",
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        default=Path("outputs/stage4_multiseed/hybrid_recommendation_diagnostics.csv"),
        help="Ranking diagnostico, incluso quando nessun candidato e' eleggibile.",
    )
    parser.add_argument(
        "--config-output",
        type=Path,
        default=Path("configs/final_config_hybrid.json"),
        help="Configurazione ibrida reviewable; il path canonico e' vietato.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configurazione di ricerca non trovata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} deve contenere un oggetto JSON.")
    return payload


def _read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} non trovato: {path}")
    return pd.read_csv(path)


def _validate_artifact_paths(arguments: argparse.Namespace) -> None:
    """Reject aliases, source overwrites, and protected final artifact trees."""

    inputs = {
        "summary": arguments.summary.resolve(),
        "raw": arguments.raw.resolve(),
        "search_config": arguments.search_config.resolve(),
    }
    outputs = {
        "config_output": arguments.config_output.resolve(),
        "recommendation_output": arguments.recommendation_output.resolve(),
        "diagnostics_output": arguments.diagnostics_output.resolve(),
    }
    for path in outputs.values():
        assert_noncanonical_config_path(path, repository_root=REPOSITORY_ROOT)

    by_destination: dict[Path, list[str]] = {}
    for label, path in outputs.items():
        by_destination.setdefault(path, []).append(label)
    collisions = {
        str(path): labels
        for path, labels in by_destination.items()
        if len(labels) > 1
    }
    if collisions:
        raise ValueError(f"Collisione tra path di output: {collisions}")

    input_by_path: dict[Path, list[str]] = {}
    for label, path in inputs.items():
        input_by_path.setdefault(path, []).append(label)
    source_collisions = {
        output_label: {
            "path": str(path),
            "inputs": input_by_path[path],
        }
        for output_label, path in outputs.items()
        if path in input_by_path
    }
    if source_collisions:
        raise ValueError(
            "Un output non puo' sovrascrivere un input: "
            f"{source_collisions}"
        )

    protected_roots = (
        (REPOSITORY_ROOT / "outputs" / "final_run").resolve(),
        (REPOSITORY_ROOT / "submission").resolve(),
    )
    protected_outputs = {
        label: str(path)
        for label, path in outputs.items()
        if any(path == root or root in path.parents for root in protected_roots)
    }
    if protected_outputs:
        raise ValueError(
            "Gli output della raccomandazione non possono essere scritti in "
            "outputs/final_run o submission: "
            f"{protected_outputs}"
        )


def main(raw_arguments: list[str] | None = None) -> int:
    """Write a recommendation and, only when safe, a validated hybrid config."""
    arguments = build_parser().parse_args(raw_arguments)
    _validate_artifact_paths(arguments)
    result = recommend_hybrid_configuration(
        _read_csv(arguments.summary, label="Summary multi-seed"),
        _read_csv(arguments.raw, label="Aggregato grezzo multi-seed"),
        _read_json(arguments.search_config),
        expected_seeds=arguments.expected_seeds,
        final_seed=arguments.seed,
    )
    if result.final_config is None:
        stale_config_exists = arguments.config_output.exists()
        recommendation = deepcopy(result.recommendation)
        recommendation.update(
            {
                "final_config_written": False,
                "config_output": str(arguments.config_output),
                "stale_config_output_exists": stale_config_exists,
            }
        )
        if stale_config_exists:
            recommendation["stale_config_warning"] = (
                "Il config-output esiste gia' ma questa esecuzione non ha prodotto "
                "una selezione eleggibile; il file preesistente non e' stato "
                "modificato e non rappresenta il risultato di questa esecuzione."
            )
        write_json(arguments.recommendation_output, recommendation)
        arguments.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
        result.diagnostics.to_csv(arguments.diagnostics_output, index=False)
        if stale_config_exists:
            print(
                "ATTENZIONE: il config-output preesistente e' obsoleto rispetto "
                "a questa esecuzione ed e' stato lasciato invariato: "
                f"{arguments.config_output}",
                file=sys.stderr,
            )
        print(
            "Nessuna configurazione finale scritta: i vincoli obbligatori non "
            "sono stati soddisfatti."
        )
        print(f"Diagnostica: {arguments.diagnostics_output}")
        return 2

    # The pure recommendation only claims generation.  Persist the validated
    # config first, then record the stronger write claim in the audit artifact.
    write_json(arguments.config_output, result.final_config)
    recommendation = deepcopy(result.recommendation)
    recommendation.update(
        {
            "final_config_written": True,
            "config_output": str(arguments.config_output),
            "stale_config_output_exists": False,
        }
    )
    write_json(arguments.recommendation_output, recommendation)
    arguments.diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    result.diagnostics.to_csv(arguments.diagnostics_output, index=False)
    print(f"Raccomandazione: {arguments.recommendation_output}")
    print(f"Configurazione ibrida: {arguments.config_output}")
    print(
        "Nota: la proxy locale non equivale alla MIA ufficiale nascosta; "
        "la valutazione ufficiale resta pendente."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
