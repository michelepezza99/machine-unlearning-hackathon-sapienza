"""Generate progressive search stages from real preceding-stage evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.progressive import (  # noqa: E402
    generate_stage2_config,
    generate_stage3_config,
    generate_stage4_config,
    semantic_candidate_fingerprint,
)
from machine_unlearning.search import validate_search_config  # noqa: E402


CANONICAL_FINAL_CONFIG = REPOSITORY_ROOT / "configs/final_config.json"
PROTECTED_OUTPUT_DIRECTORIES = (
    REPOSITORY_ROOT / "outputs/final_run",
    REPOSITORY_ROOT / "submission",
)
SHARED_SEMANTIC_SETTINGS = (
    "schema_version",
    "seed",
    "validation_fraction",
    "evaluation_batch_size",
    "utility_floor_ratio",
    "add_gradient_ascent_variants",
    "retraining",
    "fisher",
    "common_candidate",
)


def _add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_results: Path,
    default_template: Path,
    default_output: Path,
) -> None:
    parser.add_argument(
        "--results",
        type=Path,
        default=default_results,
        help="search_comparison.csv reale dello stage precedente.",
    )
    parser.add_argument(
        "--template-config",
        type=Path,
        default=default_template,
        help="Configurazione da cui preservare i valori condivisi.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Configurazione di ricerca generata.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the deterministic Stage 2/3/4 generation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Genera configurazioni progressive usando soltanto evidenza reale valida "
            "che supera l'utility floor."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    stage2 = subparsers.add_parser(
        "stage2", help="Genera il raffinamento strutturale da Stage 1."
    )
    _add_common_arguments(
        stage2,
        default_results=Path("outputs/stage1_full/search_comparison.csv"),
        default_template=Path("configs/search_stage1_coarse.json"),
        default_output=Path("configs/search_stage2_refinement.json"),
    )

    stage3 = subparsers.add_parser(
        "stage3", help="Genera il raffinamento GA/repair da Stage 2."
    )
    _add_common_arguments(
        stage3,
        default_results=Path("outputs/stage2_refinement/search_comparison.csv"),
        default_template=Path("configs/search_stage2_refinement.json"),
        default_output=Path("configs/search_stage3_finalists.json"),
    )

    stage4 = subparsers.add_parser(
        "stage4", help="Seleziona i finalisti ibridi da valutare su piu' seed."
    )
    _add_common_arguments(
        stage4,
        default_results=Path("outputs/stage3_refinement/search_comparison.csv"),
        default_template=Path("configs/search_stage3_finalists.json"),
        default_output=Path("configs/search_stage4_multiseed.json"),
    )
    stage4.add_argument(
        "--finalist-count",
        type=int,
        default=4,
        help="Numero di configurazioni ibride distinte da conservare.",
    )
    return parser


def _load_mapping(path: Path, *, description: str = "Configurazione") -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} non trovata: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"{description} deve contenere un oggetto JSON: {path}")
    return dict(payload)


def _load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Risultati dello stage precedente non trovati: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Il file dei risultati e' vuoto: {path}")
    return frame


def _canonical_json_value(value: Any, *, context: str) -> Any:
    """Normalize JSON-compatible values so equivalent numeric types compare equal."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{context} contiene un numero non finito: {value!r}.")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item, context=f"{context}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _canonical_json_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{context} contiene un valore JSON non supportato: {value!r}.")


def _shared_setting_identity(config: Mapping[str, Any], field: str) -> str:
    if field not in config:
        raise KeyError(f"Configurazione priva dell'impostazione condivisa {field!r}.")
    value = config[field]
    if field == "common_candidate":
        if not isinstance(value, Mapping):
            raise TypeError("common_candidate deve essere un mapping.")
        common_candidate = dict(value)
        # build_effective_search_config injects the authoritative global floor
        # here; it is already compared independently above.
        common_candidate.pop("utility_floor_ratio", None)
        return semantic_candidate_fingerprint({}, common_candidate)
    if field == "retraining":
        if not isinstance(value, Mapping):
            raise TypeError("retraining deve essere un mapping.")
        value = dict(value)
        if isinstance(value.get("optimizer"), str):
            value["optimizer"] = value["optimizer"].lower()
    normalized = _canonical_json_value(value, context=field)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _shared_setting_mismatches(
    evidence_config: Mapping[str, Any], template_config: Mapping[str, Any]
) -> list[str]:
    return [
        field
        for field in SHARED_SEMANTIC_SETTINGS
        if _shared_setting_identity(evidence_config, field)
        != _shared_setting_identity(template_config, field)
    ]


def _validate_evidence_bundle(
    results_path: Path,
    template_path: Path,
    template_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require completed full-run provenance compatible with the template."""
    metadata_path = results_path.parent / "search_metadata.json"
    effective_path = results_path.parent / "effective_search_config.json"
    metadata = _load_mapping(metadata_path, description="Metadati della ricerca")
    if metadata.get("status") != "completed":
        raise ValueError(
            "search_metadata.json deve dichiarare status='completed'; "
            f"ricevuto {metadata.get('status')!r}."
        )
    if metadata.get("mode") != "full":
        raise ValueError(
            "La generazione progressiva richiede evidenza mode='full'; "
            f"ricevuto {metadata.get('mode')!r}."
        )
    effective_config = _load_mapping(
        effective_path,
        description="Configurazione effettiva della ricerca",
    )
    validate_search_config(template_config)
    validate_search_config(effective_config)

    embedded_config = metadata.get("effective_search_config")
    embedded_checked = isinstance(embedded_config, Mapping)
    if embedded_checked:
        embedded_mismatches = _shared_setting_mismatches(
            embedded_config, effective_config
        )
        if embedded_mismatches:
            raise ValueError(
                "search_metadata.json ed effective_search_config.json discordano "
                f"nelle impostazioni condivise: {embedded_mismatches}."
            )

    mismatches = _shared_setting_mismatches(effective_config, template_config)
    if mismatches:
        raise ValueError(
            "La configurazione effettiva che ha prodotto i risultati non coincide "
            f"col template nelle impostazioni condivise: {mismatches}."
        )
    return {
        "validated": True,
        "producer_status": "completed",
        "producer_mode": "full",
        "results": str(results_path),
        "search_metadata": str(metadata_path),
        "effective_search_config": str(effective_path),
        "template_config": str(template_path),
        "shared_settings_checked": list(SHARED_SEMANTIC_SETTINGS),
        "shared_settings_match": True,
        "metadata_embedded_config_checked": embedded_checked,
    }


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def _assert_safe_output_path(
    output_path: Path,
    *,
    results_path: Path,
    template_path: Path,
) -> None:
    resolved_output = output_path.resolve()
    protected_files = {
        results_path.resolve(): "file dei risultati",
        template_path.resolve(): "configurazione template",
        CANONICAL_FINAL_CONFIG.resolve(): "configurazione finale canonica",
    }
    if resolved_output in protected_files:
        raise ValueError(
            "Output rifiutato: coincide con "
            f"{protected_files[resolved_output]} ({resolved_output})."
        )
    protected_directory = next(
        (
            directory
            for directory in PROTECTED_OUTPUT_DIRECTORIES
            if _is_within(resolved_output, directory)
        ),
        None,
    )
    if protected_directory is not None:
        raise ValueError(
            "Output rifiutato dentro una directory finale protetta: "
            f"{protected_directory}."
        )


def _write_config(
    path: Path,
    payload: Mapping[str, Any],
    *,
    results_path: Path,
    template_path: Path,
) -> None:
    _assert_safe_output_path(
        path,
        results_path=results_path,
        template_path=template_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(raw_arguments: Sequence[str] | None = None) -> int:
    """Generate the requested stage and report its exact candidate budget."""
    arguments = build_parser().parse_args(raw_arguments)
    _assert_safe_output_path(
        arguments.output,
        results_path=arguments.results,
        template_path=arguments.template_config,
    )
    template = _load_mapping(
        arguments.template_config,
        description="Configurazione template",
    )
    results = _load_results(arguments.results)
    evidence_validation = _validate_evidence_bundle(
        arguments.results,
        arguments.template_config,
        template,
    )
    common_arguments = {
        "source_results": str(arguments.results),
    }
    if arguments.stage == "stage2":
        generated = generate_stage2_config(template, results, **common_arguments)
    elif arguments.stage == "stage3":
        generated = generate_stage3_config(template, results, **common_arguments)
    elif arguments.stage == "stage4":
        generated = generate_stage4_config(
            template,
            results,
            finalist_count=arguments.finalist_count,
            **common_arguments,
        )
    else:  # pragma: no cover - argparse enforces the available subcommands
        raise ValueError(f"Stage non supportato: {arguments.stage!r}")
    generated["progressive_generation"][
        "evidence_validation"
    ] = evidence_validation
    validate_search_config(generated)
    _write_config(
        arguments.output,
        generated,
        results_path=arguments.results,
        template_path=arguments.template_config,
    )
    stage = generated["progressive_generation"]["stage"]
    print(f"Configurazione Stage {stage} scritta in:")
    print(arguments.output)
    print(f"Candidati base: {len(generated['candidates'])}")
    print(
        "Varianti GA automatiche: "
        f"{generated['add_gradient_ascent_variants']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
