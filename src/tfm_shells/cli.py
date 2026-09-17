from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TFM shell research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    architect = subparsers.add_parser("architect", help="Train the flow-matching Architect")
    architect.add_argument("--config", default="configs/architect.yaml")

    engineer = subparsers.add_parser("engineer", help="Train the engineer surrogate")
    engineer.add_argument("--config", default="configs/engineer.yaml")

    sample = subparsers.add_parser("sample", help="Run physics-guided sampling")
    sample.add_argument("--config", default="configs/sample_guided.yaml")

    benchmark = subparsers.add_parser("benchmark", help="Compare ODE step counts using paired initial noise")
    benchmark.add_argument("--config", default="configs/sample_guided.yaml")
    benchmark.add_argument("--steps", nargs="+", type=int, default=[10, 20, 50, 100, 250])
    benchmark.add_argument("--reference-steps", type=int, default=1000)
    benchmark.add_argument("--output", default="artifacts/step_benchmark")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "architect":
        from tfm_shells.training.train_architect import train_architect

        train_architect(Path(args.config))
        return
    if args.command == "engineer":
        from tfm_shells.training.train_engineer import train_engineer

        train_engineer(Path(args.config))
        return
    if args.command == "sample":
        from tfm_shells.sampling.guided import run_guided_sampling

        run_guided_sampling(Path(args.config))
        return
    if args.command == "benchmark":
        from tfm_shells.sampling.benchmark import benchmark_steps

        benchmark_steps(Path(args.config), args.steps, args.reference_steps, Path(args.output))
        return
    raise ValueError(f"Unsupported command: {args.command}")
