from __future__ import annotations

import csv
from itertools import product
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_MONTE_CARLO_RUNS,
    DEFAULT_RATE_LIMIT,
    DEFAULT_SIMULATION_DURATION,
    EXPERIMENT_ALGORITHMS,
    EXPERIMENT_ARCHITECTURES,
    EXPERIMENT_ARRIVAL_RATES,
    EXPERIMENT_DISTRIBUTED_LATENCY_MS,
    EXPERIMENT_FAILURE_RATE,
    EXPERIMENT_FAIL_CLOSED,
    EXPERIMENT_TRAFFIC_PATTERNS,
    EXPERIMENT_USERS,
    SimulationConfig,
)
from .metrics import calculate_metrics
from .simulator import run_simulation


# These are the configuration columns that are written to both CSV files.
CONFIGURATION_FIELDS = (
    "users",
    "arrival_rate",
    "traffic_pattern",
    "algorithm",
    "architecture",
    "distributed_latency",
    "failure_rate",
    "fail_closed",
    "rate_limit",
    "simulation_duration",
)


# These are the metrics produced by one simulation.
METRIC_NAMES = (
    "throughput",
    "acceptance_rate",
    "rejection_rate",
    "p95_latency",
    "jain_fairness",
    "limit_overshoot",
)


# Columns in results_raw.csv.
RAW_FIELDS = (
    "experiment_id",
    "run_id",
    "seed",
    *CONFIGURATION_FIELDS,
    "total_requests",
    "accepted_requests",
    "rejected_requests",
    *METRIC_NAMES,
)


# Columns in results.csv.
AGGREGATED_FIELDS = (
    "experiment_id",
    *CONFIGURATION_FIELDS,
    *[
        field
        for metric in METRIC_NAMES
        for field in (f"{metric}_mean", f"{metric}_std")
    ],
)


# The experiment matrix intentionally contains:
#
#   Local:
#       users             = 2 values
#       arrival_rate      = 2 values
#       traffic_pattern   = 3 values
#       algorithm         = 3 values
#       distributed-only dimensions are not varied
#
#       2 × 2 × 3 × 3 = 36
#
#   Distributed:
#       users             = 2 values
#       arrival_rate      = 2 values
#       traffic_pattern   = 3 values
#       algorithm         = 3 values
#       distributed_latency = 2 values
#       failure_rate        = 2 values
#
#       2 × 2 × 3 × 3 × 2 × 2 = 144
#
#   Total = 36 + 144 = 180 configurations.
#
# This is intentionally NOT 144. The original specification's
# "144 configurations" count was mathematically inconsistent with
# the listed dimensions once local and distributed architectures
# are handled correctly.
EXPECTED_CONFIGURATION_COUNT = 180


def generate_experiment_configurations(
    rate_limit: int = DEFAULT_RATE_LIMIT,
    simulation_duration: float = DEFAULT_SIMULATION_DURATION,
) -> list[SimulationConfig]:
    """
    Generate the complete experiment configuration matrix.

    Local architecture:
        Distributed latency and failure rate do not affect the system,
        so each is represented by one neutral value:

            distributed_latency_ms = 0
            failure_rate = 0.0

        This produces 36 local configurations.

    Distributed architecture:
        Both distributed latency and failure rate are experiment
        dimensions, producing 144 distributed configurations.

    Therefore the complete matrix contains 180 configurations.
    """
    configurations: list[SimulationConfig] = []

    experiment_id = 1

    # ------------------------------------------------------------------
    # Local configurations
    # ------------------------------------------------------------------
    #
    # There is no reason to run four copies of every local configuration
    # just because distributed_latency and failure_rate have values in
    # the global experiment configuration.
    #
    # Local architecture does not use either setting operationally.
    #
    for (
        users,
        arrival_rate,
        traffic_pattern,
        algorithm,
    ) in product(
        EXPERIMENT_USERS,
        EXPERIMENT_ARRIVAL_RATES,
        EXPERIMENT_TRAFFIC_PATTERNS,
        EXPERIMENT_ALGORITHMS,
    ):
        configurations.append(
            SimulationConfig(
                users=users,
                arrival_rate=arrival_rate,
                traffic_pattern=traffic_pattern,
                algorithm=algorithm,
                architecture="local",
                distributed_latency_ms=0,
                failure_rate=0.0,
                fail_closed=EXPERIMENT_FAIL_CLOSED,
                rate_limit=rate_limit,
                simulation_duration=simulation_duration,
                seed=experiment_id,
            )
        )

        experiment_id += 1

    # ------------------------------------------------------------------
    # Distributed configurations
    # ------------------------------------------------------------------
    #
    # For distributed architecture, latency and failure rate are
    # meaningful dimensions and therefore every combination is tested.
    #
    for (
        users,
        arrival_rate,
        traffic_pattern,
        algorithm,
        distributed_latency_ms,
        failure_rate,
    ) in product(
        EXPERIMENT_USERS,
        EXPERIMENT_ARRIVAL_RATES,
        EXPERIMENT_TRAFFIC_PATTERNS,
        EXPERIMENT_ALGORITHMS,
        EXPERIMENT_DISTRIBUTED_LATENCY_MS,
        EXPERIMENT_FAILURE_RATE,
    ):
        configurations.append(
            SimulationConfig(
                users=users,
                arrival_rate=arrival_rate,
                traffic_pattern=traffic_pattern,
                algorithm=algorithm,
                architecture="distributed",
                distributed_latency_ms=distributed_latency_ms,
                failure_rate=failure_rate,
                fail_closed=EXPERIMENT_FAIL_CLOSED,
                rate_limit=rate_limit,
                simulation_duration=simulation_duration,
                seed=experiment_id,
            )
        )

        experiment_id += 1

    if len(configurations) != EXPECTED_CONFIGURATION_COUNT:
        raise AssertionError(
            f"Expected exactly {EXPECTED_CONFIGURATION_COUNT} "
            f"configurations, got {len(configurations)}"
        )

    return configurations


def derive_run_seed(
    base_seed: int,
    experiment_index: int,
    run_id: int,
) -> int:
    """
    Derive a deterministic unique seed for one experiment run.

    SeedSequence gives us independent-looking seeds while keeping the
    entire experiment suite reproducible from one base seed.
    """
    sequence = np.random.SeedSequence(
        entropy=base_seed,
        spawn_key=(experiment_index, run_id),
    )

    return int(
        sequence.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )


def _raw_row(
    experiment_id: int,
    run_id: int,
    seed: int,
    config: SimulationConfig,
    metrics: dict[str, float],
) -> dict[str, object]:
    """Build one row for results_raw.csv."""
    row: dict[str, object] = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "seed": seed,
    }

    row.update(config.csv_dict())

    row.update(
        {
            "total_requests": int(metrics["total_requests"]),
            "accepted_requests": int(metrics["accepted_requests"]),
            "rejected_requests": int(metrics["rejected_requests"]),
        }
    )

    for metric in METRIC_NAMES:
        row[metric] = metrics[metric]

    return row


def _aggregate_rows(
    configuration: SimulationConfig,
    experiment_id: int,
    metric_rows: list[dict[str, float]],
) -> dict[str, object]:
    """
    Aggregate the Monte Carlo metrics for one configuration.

    Population standard deviation (ddof=0) is used consistently.
    """
    if not metric_rows:
        raise ValueError("Cannot aggregate an empty set of metric rows")

    row: dict[str, object] = {
        "experiment_id": experiment_id,
    }

    row.update(configuration.csv_dict())

    for metric in METRIC_NAMES:
        values = np.array(
            [result[metric] for result in metric_rows],
            dtype=float,
        )

        row[f"{metric}_mean"] = float(np.mean(values))

        # Population standard deviation.
        row[f"{metric}_std"] = float(
            np.std(
                values,
                ddof=0,
            )
        )

    return row


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    """Write rows to a CSV file."""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
        )

        writer.writeheader()
        writer.writerows(rows)


def run_experiments(
    base_seed: int = 42,
    simulation_duration: float = DEFAULT_SIMULATION_DURATION,
    output_dir: str | Path = "results",
    monte_carlo_runs: int = DEFAULT_MONTE_CARLO_RUNS,
    rate_limit: int = DEFAULT_RATE_LIMIT,
) -> tuple[Path, Path, int, int]:
    """
    Run the complete Monte Carlo experiment suite.

    Returns:
        (
            raw_results_path,
            aggregate_results_path,
            configuration_count,
            total_run_count,
        )
    """
    if monte_carlo_runs <= 0:
        raise ValueError(
            "monte_carlo_runs must be positive"
        )

    configurations = generate_experiment_configurations(
        rate_limit=rate_limit,
        simulation_duration=simulation_duration,
    )

    output_path = Path(output_dir)

    raw_path = output_path / "results_raw.csv"
    aggregate_path = output_path / "results.csv"

    raw_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []

    total_runs = (
        len(configurations)
        * monte_carlo_runs
    )

    print()
    print("=== Experiment Suite ===")
    print(
        f"Configurations:              "
        f"{len(configurations)}"
    )
    print(
        f"Monte Carlo runs/config:     "
        f"{monte_carlo_runs}"
    )
    print(
        f"Total simulation runs:       "
        f"{total_runs}"
    )
    print(
        f"Simulation duration:         "
        f"{simulation_duration:g} sec"
    )
    print(
        f"Base seed:                   "
        f"{base_seed}"
    )
    print()

    for configuration_index, configuration in enumerate(
        configurations,
        start=1,
    ):
        print(
            f"Running configuration "
            f"{configuration_index}/{len(configurations)}"
        )

        metric_rows: list[dict[str, float]] = []

        for run_id in range(
            1,
            monte_carlo_runs + 1,
        ):
            # Avoid printing thousands of lines while still showing
            # useful progress.
            if (
                run_id == 1
                or run_id == monte_carlo_runs
                or run_id % 10 == 0
            ):
                print(
                    f"  Monte Carlo run "
                    f"{run_id}/{monte_carlo_runs}"
                )

            seed = derive_run_seed(
                base_seed=base_seed,
                experiment_index=configuration_index,
                run_id=run_id,
            )

            # The configuration itself has an experiment-level seed,
            # but each Monte Carlo run gets its own deterministic seed.
            run_config = configuration.with_seed(seed)

            # Run exactly one simulation.
            result = run_simulation(run_config)

            # Metrics are deliberately calculated outside the simulator.
            metrics = calculate_metrics(
                result,
                run_config,
            )

            metric_rows.append(metrics)

            raw_rows.append(
                _raw_row(
                    experiment_id=configuration_index,
                    run_id=run_id,
                    seed=seed,
                    config=run_config,
                    metrics=metrics,
                )
            )

        aggregate_rows.append(
            _aggregate_rows(
                configuration=configuration,
                experiment_id=configuration_index,
                metric_rows=metric_rows,
            )
        )

    # Write only after all simulations have completed. This keeps the
    # simulation loop simple and makes the resulting CSV files complete.
    _write_csv(
        raw_path,
        raw_rows,
        RAW_FIELDS,
    )

    _write_csv(
        aggregate_path,
        aggregate_rows,
        AGGREGATED_FIELDS,
    )

    return (
        raw_path,
        aggregate_path,
        len(configurations),
        total_runs,
    )