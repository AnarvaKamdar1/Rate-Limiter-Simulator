from __future__ import annotations

from collections import deque

import numpy as np

from .config import SimulationConfig
from .simulator import SimulationResult


def throughput(
    accepted_requests: int,
    simulation_duration: float,
) -> float:
    """Calculate actual accepted throughput in requests/second."""
    if simulation_duration <= 0:
        raise ValueError("simulation_duration must be positive")

    return accepted_requests / simulation_duration


def acceptance_rate(
    accepted_requests: int,
    total_requests: int,
) -> float:
    """Calculate accepted requests divided by all requests."""
    if total_requests == 0:
        return 0.0

    return accepted_requests / total_requests


def rejection_rate(
    rejected_requests: int,
    total_requests: int,
) -> float:
    """Calculate rejected requests divided by all requests."""
    if total_requests == 0:
        return 0.0

    return rejected_requests / total_requests


def p95_decision_latency(result: SimulationResult) -> float:
    """Return the 95th percentile latency of accepted requests in seconds.

    Latency is measured from request arrival until its admission/service
    decision time. For queue-based algorithms such as leaky bucket, this
    includes time spent waiting in the queue.
    """
    if not result.events:
        return 0.0

    # Rejected requests do not incur queueing/decision latency in the
    # limiter.  Measure latency only for requests that were actually
    # admitted.  This makes the metric meaningful for algorithms such as
    # leaky bucket, where accepted requests may wait for service.
    accepted_latencies = [
        max(0.0, event.decision_time - event.arrival_time)
        for event in result.events
        if event.allowed
    ]

    if not accepted_latencies:
        return 0.0

    latencies = np.array(accepted_latencies, dtype=float)

    # ``midpoint`` preserves the percentile convention used by the existing
    # test data (e.g. the 10-value example gives 0.095 exactly).
    return float(np.percentile(latencies, 95, method="midpoint"))


def jain_fairness(
    accepted_by_user: dict[int, int],
    user_count: int,
) -> float:
    """Calculate Jain's fairness index over all configured users."""
    if user_count <= 0:
        raise ValueError("user_count must be positive")

    values = np.array(
        [
            float(accepted_by_user.get(user_id, 0))
            for user_id in range(user_count)
        ],
        dtype=float,
    )

    numerator = float(values.sum() ** 2)
    denominator = float(user_count * np.sum(values**2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def limit_overshoot(
    admitted_times: list[float],
    limit: int,
    window_size: float = 1.0,
) -> float:
    """Calculate maximum one-window limit overshoot as a percentage.

    The measurement is:

        max(0, admitted_in_any_window - limit) / limit * 100

    Windows are evaluated using the observed request timestamps and are
    interpreted as half-open intervals [t, t + window_size).
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    if not admitted_times:
        return 0.0

    times = sorted(admitted_times)

    # Two-pointer sliding window.  The interval is [t, t + window_size),
    # matching the semantics of the rate-limit window.
    left = 0
    maximum = 0

    for right, current_time in enumerate(times):
        while left <= right and times[left] <= current_time - window_size:
            left += 1

        maximum = max(maximum, right - left + 1)

    overshoot = max(0, maximum - limit)
    return overshoot / limit * 100.0


def calculate_metrics(
    result: SimulationResult,
    config: SimulationConfig,
) -> dict[str, float]:
    """Calculate all requested metrics for a simulation result."""
    total = result.total_requests
    accepted = result.accepted_requests
    rejected = result.rejected_requests

    return {
        "total_requests": float(total),
        "accepted_requests": float(accepted),
        "rejected_requests": float(rejected),
        "throughput": throughput(
            accepted,
            config.simulation_duration,
        ),
        "acceptance_rate": acceptance_rate(
            accepted,
            total,
        ),
        "rejection_rate": rejection_rate(
            rejected,
            total,
        ),
        "p95_latency": p95_decision_latency(result),
        "jain_fairness": jain_fairness(
            result.accepted_by_user,
            config.users,
        ),
        "limit_overshoot": limit_overshoot(
            result.admitted_times,
            config.rate_limit,
            window_size=1.0,
        ),
    }