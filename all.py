from __future__ import annotations

import argparse

from src.config import (
    DEFAULT_RATE_LIMIT,
    DEFAULT_SIMULATION_DURATION,
)
from src.experiments import run_experiments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete rate limiter experiment suite."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed used to deterministically derive every run seed.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_SIMULATION_DURATION,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for raw and aggregated CSV files.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Monte Carlo runs per configuration. Default: 50.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_RATE_LIMIT,
        help="Rate limit used by every experiment configuration.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    raw_path, aggregate_path, configuration_count, run_count = run_experiments(
        base_seed=args.seed,
        simulation_duration=args.duration,
        output_dir=args.output_dir,
        monte_carlo_runs=args.runs,
        rate_limit=args.rate_limit,
    )

    print()
    print("=== Experiment Suite Complete ===")
    print(f"configurations completed: {configuration_count}")
    print(f"total simulation runs:    {run_count}")
    print(f"raw results:              {raw_path}")
    print(f"aggregated results:       {aggregate_path}")


if __name__ == "__main__":
    main()