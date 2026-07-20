"""Entry point riproducibile per la submission finale."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from machine_unlearning.workflow import run_final_workflow


def build_parser() -> argparse.ArgumentParser:
    """Definisce la piccola interfaccia a riga di comando del workflow finale."""
    parser = argparse.ArgumentParser(
        description=(
            "Esegue il metodo di machine unlearning fissato e genera una "
            "submission validata."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_run"))
    parser.add_argument("--submission-dir", type=Path, default=Path("submission"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/final_config.json")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sovrascrive il seed della configurazione finale.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device PyTorch: auto, cpu, cuda o un device esplicito.",
    )
    return parser


def main() -> int:
    """Esegue il workflow e stampa un riepilogo compatto."""
    arguments = build_parser().parse_args()
    summary = run_final_workflow(
        data_dir=arguments.data_dir,
        output_dir=arguments.output_dir,
        submission_dir=arguments.submission_dir,
        config_path=arguments.config,
        seed_override=arguments.seed,
        device_name=arguments.device,
    )
    print("\nSubmission pronta")
    print(f"Metodo: {summary.method}")
    print(f"Device: {summary.device}")
    print(f"Directory: {summary.submission_dir}")
    print(f"Tempo dichiarato: {math.ceil(summary.execution_time_seconds)} s")
    print(
        "Metriche locali post-hoc: "
        f"P@10={summary.metrics['validation_precision_at_10']:.6f}, "
        f"BCE validation={summary.metrics['validation_bce']:.6f}, "
        f"BCE forget={summary.metrics['forget_bce']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
