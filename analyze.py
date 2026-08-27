from __future__ import annotations

import argparse

from src.analysis import (
    METRIC_DIRECTIONS,
    SUPPORTED_METRICS,
    generate_all_plots,
    generate_plot,
    load_results,
    print_algorithm_comparison,
)


PLOT_NAMES = (
    "algorithm_latency",
    "algorithm_rejection",
    "algorithm_fairness",
    "architecture_latency",
    "failure_rejection",
    "distributed_latency",
    "traffic_throughput",
    "all",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Query and plot aggregated rate limiter results."
    )

    parser.add_argument(
        "--results",
        default="results/results.csv",
        help="Path to the aggregated results CSV.",
    )

    parser.add_argument("--users", type=int)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--traffic")
    parser.add_argument("--algorithm")
    parser.add_argument("--architecture")

    # CLI uses the friendly name "distributed-latency".
    # The CSV column is also named "distributed_latency".
    parser.add_argument("--distributed-latency", type=float)

    parser.add_argument("--failure-rate", type=float)

    parser.add_argument(
        "--metric",
        choices=SUPPORTED_METRICS,
        default="p95_latency",
    )

    parser.add_argument(
        "--plot",
        choices=PLOT_NAMES,
        help="Generate a plot instead of printing an algorithm comparison.",
    )

    parser.add_argument(
        "--plot-dir",
        default="plots",
        help="Directory where generated plots are saved.",
    )

    return parser


def main() -> None:
    """Run the analysis command-line interface."""
    args = build_parser().parse_args()

    # These names must match the actual column names in results.csv.
    #
    # Note:
    # SimulationConfig internally uses distributed_latency_ms,
    # but CSV output intentionally uses distributed_latency.
    filters = {
        "users": args.users,
        "arrival_rate": args.arrival_rate,
        "traffic_pattern": args.traffic,
        "algorithm": args.algorithm,
        "architecture": args.architecture,
        "distributed_latency": args.distributed_latency,
        "failure_rate": args.failure_rate,
    }

    data = load_results(args.results)
    filtered = data

    for column, value in filters.items():
        if value is not None:
            filtered = filtered[filtered[column] == value]

    if filtered.empty:
        print("No matching configurations were found.")
        return

    if args.plot:
        if args.plot == "all":
            paths = generate_all_plots(
                filtered,
                output_dir=args.plot_dir,
            )

            print("Generated plots:")

            for path in paths:
                print(f"  {path}")
        else:
            path = generate_plot(
                filtered,
                plot_name=args.plot,
                output_dir=args.plot_dir,
            )

            print(f"Generated plot: {path}")

        return

    print_algorithm_comparison(
        filtered,
        metric=args.metric,
    )

    direction = METRIC_DIRECTIONS[args.metric]

    print()
    print(f"Metric direction: {direction}")


if __name__ == "__main__":
    main()