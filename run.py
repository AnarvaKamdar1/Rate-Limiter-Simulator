from __future__ import annotations

import argparse

from src.config import (
    ALGORITHMS,
    ARCHITECTURES,
    TRAFFIC_PATTERNS,
    DEFAULT_DISTRIBUTED_LATENCY_MS,
    DEFAULT_FAILURE_RATE,
    DEFAULT_RATE_LIMIT,
    DEFAULT_SIMULATION_DURATION,
    DEFAULT_USERS,
    DEFAULT_ARRIVAL_RATE,
    SimulationConfig,
)
from src.metrics import calculate_metrics
from src.simulator import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one rate limiter simulation."
    )

    parser.add_argument("--users", type=int, default=DEFAULT_USERS)
    parser.add_argument("--arrival-rate", type=float, default=DEFAULT_ARRIVAL_RATE)
    parser.add_argument(
        "--traffic",
        choices=TRAFFIC_PATTERNS,
        default="constant",
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        default="sliding_window",
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURES,
        default="local",
    )
    parser.add_argument(
        "--distributed-latency",
        type=float,
        default=DEFAULT_DISTRIBUTED_LATENCY_MS,
        help="Distributed-state latency in milliseconds.",
    )
    parser.add_argument(
        "--failure-rate",
        type=float,
        default=DEFAULT_FAILURE_RATE,
        help="Distributed-state failure probability, e.g. 0.01 for 1%%.",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=DEFAULT_RATE_LIMIT,
        help="Requests allowed per one-second rate-limiting window.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_SIMULATION_DURATION,
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser


def print_summary(config: SimulationConfig, metrics: dict[str, float]) -> None:
    print("=== Configuration ===")
    print(f"users:                 {config.users}")
    print(f"arrival_rate:          {config.arrival_rate:g}")
    print(f"traffic_pattern:       {config.traffic_pattern}")
    print(f"algorithm:             {config.algorithm}")
    print(f"architecture:          {config.architecture}")
    print(f"distributed_latency:   {config.distributed_latency_ms:g} ms")
    print(f"failure_rate:          {config.failure_rate:.2%}")
    print(f"fail_closed:           {config.fail_closed}")
    print(f"rate_limit:            {config.rate_limit} requests/sec")
    print(f"duration:              {config.simulation_duration:g} sec")
    print(f"seed:                  {config.seed}")

    print()
    print("=== Simulation Summary ===")
    print(f"total_requests:        {metrics['total_requests']:.0f}")
    print(f"accepted:              {metrics['accepted_requests']:.0f}")
    print(f"rejected:              {metrics['rejected_requests']:.0f}")
    print(f"throughput:            {metrics['throughput']:.3f} requests/sec")
    print(f"acceptance_rate:       {metrics['acceptance_rate']:.3%}")
    print(f"rejection_rate:        {metrics['rejection_rate']:.3%}")
    print(f"p95_latency:           {metrics['p95_latency'] * 1000:.3f} ms")
    print(f"jain_fairness:         {metrics['jain_fairness']:.4f}")
    print(f"limit_overshoot:       {metrics['limit_overshoot']:.3f}%")


def main() -> None:
    args = build_parser().parse_args()

    config = SimulationConfig(
        users=args.users,
        arrival_rate=args.arrival_rate,
        traffic_pattern=args.traffic,
        algorithm=args.algorithm,
        architecture=args.architecture,
        distributed_latency_ms=args.distributed_latency,
        failure_rate=args.failure_rate,
        fail_closed=True,
        rate_limit=args.rate_limit,
        simulation_duration=args.duration,
        seed=args.seed,
    )

    result = run_simulation(config)
    metrics = calculate_metrics(result, config)

    print_summary(config, metrics)


if __name__ == "__main__":
    main()