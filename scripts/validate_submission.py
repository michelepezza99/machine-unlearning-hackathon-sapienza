"""Valida una submission indipendentemente da `main.py`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from machine_unlearning.submission import validate_submission  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Definisce gli argomenti del validator autonomo."""
    parser = argparse.ArgumentParser(description="Valida i tre file di submission.")
    parser.add_argument("--submission-dir", type=Path, default=Path("submission"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    return parser


def main() -> int:
    """Esegue la validazione e stampa un riepilogo essenziale."""
    arguments = build_parser().parse_args()
    result = validate_submission(arguments.submission_dir, data_dir=arguments.data_dir)
    print("Submission valida")
    print(f"File: {', '.join(result['files'])}")
    print(f"Validation ID: {result['validation_id_count']}")
    print(f"Tempo dichiarato: {result['declared_execution_time_seconds']} s")
    print(f"Inferenza su dati reali: {result['inference_checked']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
