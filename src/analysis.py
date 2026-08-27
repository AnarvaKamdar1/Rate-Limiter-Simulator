from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUPPORTED_METRICS = (
    "throughput",
    "acceptance_rate",
    "rejection_rate",
    "p95_latency",
    "jain_fairness",
    "limit_overshoot",
)

METRIC_DIRECTIONS = {
    "throughput": "higher is better",
    "acceptance_rate": "higher is better",
    "rejection_rate": "lower is better",
    "p95_latency": "lower is better",
    "jain_fairness": "higher is better",
    "limit_overshoot": "lower is better",
}


def load_results(path: str | Path = "results/results.csv") -> pd.DataFrame:
    """Load aggregated experiment results."""
    result_path = Path(path)

    if not result_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {result_path}. "
            "Run all.py first."
        )

    return pd.read_csv(result_path)


def filter_results(
    data: pd.DataFrame,
    **filters: object,
) -> pd.DataFrame:
    """Filter a result dataframe; None values act as wildcards."""
    filtered = data

    for column, value in filters.items():
        if value is not None:
            if column not in filtered.columns:
                raise KeyError(f"Unknown result column: {column}")
            filtered = filtered[filtered[column] == value]

    return filtered


def _display_value(metric: str, value: float) -> str:
    if metric == "p95_latency":
        return f"{value * 1000:.3f} ms"
    if metric in {"acceptance_rate", "rejection_rate"}:
        return f"{value:.2%}"
    if metric == "limit_overshoot":
        return f"{value:.3f}%"
    if metric == "jain_fairness":
        return f"{value:.4f}"
    return f"{value:.3f}"


def print_algorithm_comparison(
    data: pd.DataFrame,
    metric: str,
) -> None:
    """Print a beginner-friendly algorithm comparison."""
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")

    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    if mean_column not in data.columns:
        raise KeyError(f"Missing result column: {mean_column}")

    comparison = (
        data.groupby("algorithm", as_index=False)[
            [mean_column, std_column]
        ]
        .mean()
    )

    ascending = METRIC_DIRECTIONS[metric] == "lower is better"
    comparison = comparison.sort_values(
        by=mean_column,
        ascending=ascending,
    )

    print("## Algorithm comparison")
    print()

    for _, row in comparison.iterrows():
        mean_value = _display_value(metric, float(row[mean_column]))
        std_value = _display_value(metric, float(row[std_column]))

        print(
            f"{row['algorithm']:<18} "
            f"{mean_value} ± {std_value}"
        )

    winner = comparison.iloc[0]["algorithm"]

    print()
    print(f"Winner: {winner}")


def _save_figure(
    figure,
    output_dir: str | Path,
    filename: str,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    path = output_path / filename

    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)

    return path


def _algorithm_plot(
    data: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: str | Path,
) -> Path:
    grouped = (
        data.groupby("algorithm")[f"{metric}_mean"]
        .mean()
        .sort_index()
    )

    figure, axis = plt.subplots(figsize=(8, 5))

    grouped.plot(
        kind="bar",
        ax=axis,
        color=["#4472C4", "#70AD47", "#ED7D31"],
    )

    axis.set_title(title)
    axis.set_xlabel("Algorithm")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=0.25)

    return _save_figure(figure, output_dir, filename)


def generate_plot(
    data: pd.DataFrame,
    plot_name: str,
    output_dir: str | Path = "plots",
) -> Path:
    """Generate one named analysis plot."""
    if data.empty:
        raise ValueError("Cannot plot an empty dataframe.")

    if plot_name == "algorithm_latency":
        return _algorithm_plot(
            data,
            metric="p95_latency",
            title="Algorithm vs P95 Decision Latency",
            ylabel="P95 latency (seconds)",
            filename="algorithm_vs_p95_latency.png",
            output_dir=output_dir,
        )

    if plot_name == "algorithm_rejection":
        return _algorithm_plot(
            data,
            metric="rejection_rate",
            title="Algorithm vs Rejection Rate",
            ylabel="Rejection rate",
            filename="algorithm_vs_rejection_rate.png",
            output_dir=output_dir,
        )

    if plot_name == "algorithm_fairness":
        return _algorithm_plot(
            data,
            metric="jain_fairness",
            title="Algorithm vs Jain Fairness",
            ylabel="Jain fairness",
            filename="algorithm_vs_fairness.png",
            output_dir=output_dir,
        )

    if plot_name == "architecture_latency":
        grouped = (
            data.groupby("architecture")["p95_latency_mean"]
            .mean()
        )

        figure, axis = plt.subplots(figsize=(8, 5))
        grouped.plot(kind="bar", ax=axis, color="#5B9BD5")

        axis.set_title("Architecture vs P95 Decision Latency")
        axis.set_xlabel("Architecture")
        axis.set_ylabel("P95 latency (seconds)")
        axis.grid(axis="y", alpha=0.25)

        return _save_figure(
            figure,
            output_dir,
            "architecture_vs_latency.png",
        )

    if plot_name == "failure_rejection":
        grouped = (
            data.groupby("failure_rate")["rejection_rate_mean"]
            .mean()
            .sort_index()
        )

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            grouped.index * 100,
            grouped.values,
            marker="o",
        )

        axis.set_title("Failure Rate vs Rejection Rate")
        axis.set_xlabel("Failure rate (%)")
        axis.set_ylabel("Rejection rate")
        axis.grid(alpha=0.25)

        return _save_figure(
            figure,
            output_dir,
            "failure_rate_vs_rejection_rate.png",
        )

    if plot_name == "distributed_latency":
        grouped = (
            data.groupby("distributed_latency")["p95_latency_mean"]
            .mean()
            .sort_index()
        )

        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(
            grouped.index,
            grouped.values * 1000,
            marker="o",
        )

        axis.set_title("Distributed Latency vs P95 Decision Latency")
        axis.set_xlabel("Distributed latency (ms)")
        axis.set_ylabel("P95 latency (ms)")
        axis.grid(alpha=0.25)

        return _save_figure(
            figure,
            output_dir,
            "distributed_latency_vs_p95_latency.png",
        )

    if plot_name == "traffic_throughput":
        grouped = (
            data.groupby("traffic_pattern")["throughput_mean"]
            .mean()
        )

        figure, axis = plt.subplots(figsize=(8, 5))
        grouped.plot(
            kind="bar",
            ax=axis,
            color="#A5A5A5",
        )

        axis.set_title("Traffic Pattern vs Throughput")
        axis.set_xlabel("Traffic pattern")
        axis.set_ylabel("Throughput (requests/sec)")
        axis.grid(axis="y", alpha=0.25)

        return _save_figure(
            figure,
            output_dir,
            "traffic_pattern_vs_throughput.png",
        )

    raise ValueError(f"Unknown plot: {plot_name}")


def generate_all_plots(
    data: pd.DataFrame,
    output_dir: str | Path = "plots",
) -> list[Path]:
    """Generate all supported plots."""
    plot_names = (
        "algorithm_latency",
        "algorithm_rejection",
        "algorithm_fairness",
        "architecture_latency",
        "failure_rejection",
        "distributed_latency",
        "traffic_throughput",
    )

    return [
        generate_plot(
            data,
            plot_name=plot_name,
            output_dir=output_dir,
        )
        for plot_name in plot_names
    ]