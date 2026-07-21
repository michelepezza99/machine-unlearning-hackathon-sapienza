"""Aggregate every per-seed hybrid search candidate and compute Pareto evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.search_aggregation import (  # noqa: E402
    LOCAL_PRIVACY_DISCLAIMER,
    aggregate_all_candidates,
    discover_search_result_files,
    infer_expected_seeds_from_root_config,
    validate_aggregation_output_paths,
    write_aggregation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit all-candidate aggregation CLI."""

    parser = argparse.ArgumentParser(
        description=(
            "Aggrega tutti i seed_*/search_comparison.csv, inclusi i fallimenti, "
            "e calcola statistiche e frontiera Pareto deterministiche."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory contenente le sottodirectory seed_<n>.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seed attesi, usati per distinguere esecuzioni mancanti e non valide.",
    )
    parser.add_argument(
        "--expected-seed-count",
        type=int,
        default=None,
        help="Numero atteso di seed, utile quando gli identificativi non sono noti.",
    )
    parser.add_argument(
        "--raw-output",
        "--combined-output",
        dest="raw_output",
        type=Path,
        default=None,
        help="Destinazione della tabella grezza combinata.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Destinazione del riepilogo per configurazione.",
    )
    parser.add_argument(
        "--pareto-output",
        type=Path,
        default=None,
        help="Destinazione dell'analisi Pareto.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Destinazione dei metadati JSON e della limitazione privacy.",
    )
    return parser


def main(raw_arguments: list[str] | None = None) -> int:
    """Aggregate configured evidence and write the four deterministic outputs."""

    arguments = build_parser().parse_args(raw_arguments)
    input_dir = arguments.input_dir
    output_paths = {
        "raw_path": arguments.raw_output
        or input_dir / "all_candidates_all_seeds.csv",
        "summary_path": arguments.summary_output
        or input_dir / "all_candidates_summary.csv",
        "pareto_path": arguments.pareto_output
        or input_dir / "all_candidates_pareto.csv",
        "metadata_path": arguments.metadata_output
        or input_dir / "all_candidates_metadata.json",
    }
    sources = discover_search_result_files(input_dir)
    validate_aggregation_output_paths(
        input_dir=input_dir,
        sources=sources,
        repository_root=REPOSITORY_ROOT,
        **output_paths,
    )
    expected_seeds = arguments.expected_seeds
    if expected_seeds is None and arguments.expected_seed_count is None:
        expected_seeds = infer_expected_seeds_from_root_config(input_dir)
    result = aggregate_all_candidates(
        input_dir,
        expected_seeds=expected_seeds,
        expected_seed_count=arguments.expected_seed_count,
    )
    destinations = write_aggregation_outputs(
        result,
        **output_paths,
    )
    print(f"Risultati grezzi: {destinations['raw']}")
    print(f"Riepilogo: {destinations['summary']}")
    print(f"Pareto: {destinations['pareto']}")
    print(f"Metadati: {destinations['metadata']}")
    print(f"Limitazione privacy: {LOCAL_PRIVACY_DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
