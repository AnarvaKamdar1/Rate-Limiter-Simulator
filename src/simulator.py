from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .config import SimulationConfig
from .generator import Request, create_requests
from .rate_limiters import Decision, create_rate_limiter
from .traffic import generate_arrival_times


@dataclass
class RequestEvent:
    """Diagnostic information for one request."""

    request_id: int
    user_id: int
    arrival_time: float
    decision_time: float
    allowed: bool
    distributed_operation_failed: bool = False
    node_id: int = 0


@dataclass
class SimulationResult:
    """Raw outcome of exactly one simulation.

    Metric calculations deliberately do not live here. This object contains
    the observations needed by metrics.py.
    """

    requests: list[Request] = field(default_factory=list)
    events: list[RequestEvent] = field(default_factory=list)
    accepted_by_user: dict[int, int] = field(default_factory=dict)
    admitted_times: list[float] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        # Normal simulations populate ``requests``.  Falling back to events
        # also keeps metrics well-defined for lightweight hand-built results
        # used by tests and diagnostics.
        return len(self.requests) if self.requests else len(self.events)

    @property
    def accepted_requests(self) -> int:
        return sum(1 for event in self.events if event.allowed)

    @property
    def rejected_requests(self) -> int:
        return sum(1 for event in self.events if not event.allowed)


class Architecture:
    """Small simulation-only architecture wrapper."""

    def __init__(
        self,
        config: SimulationConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.rng = rng

        self.rate_limiter = create_rate_limiter(
            algorithm=config.algorithm,
            limit=config.rate_limit,
            window_size=config.window_size,
            token_capacity=config.token_capacity,
            token_refill_rate=config.token_refill_rate,
            leaky_bucket_capacity=config.leaky_bucket_capacity,
            leaky_bucket_rate=config.leaky_bucket_rate,
        )

    def decide(
        self,
        request: Request,
    ) -> tuple[Decision, bool, int]:
        """Return decision, failure flag, and logical node ID."""
        node_id = request.user_id % self.config.distributed_nodes

        if self.config.architecture == "local":
            decision = self.rate_limiter.allow(
                request,
                request.arrival_time,
            )
            return decision, False, node_id

        operation_failed = bool(self.rng.random() < self.config.failure_rate)

        decision_time = (
            request.arrival_time
            + self.config.distributed_latency_ms / 1000.0
        )

        if operation_failed:
            if self.config.fail_closed:
                return (
                    Decision(False, decision_time),
                    True,
                    node_id,
                )

            # Fail-open is intentionally unsupported by the current CLI/config.
            raise RuntimeError(
                "fail_closed=False is not supported in this simulator yet."
            )

        decision = self.rate_limiter.allow(
            request,
            decision_time,
        )

        return decision, False, node_id


def run_simulation(config: SimulationConfig) -> SimulationResult:
    """Run one simulation using the configured traffic, limiter, and random seed."""
    rng = np.random.default_rng(config.seed)

    arrival_times = generate_arrival_times(
        pattern=config.traffic_pattern,
        arrival_rate=config.arrival_rate,
        duration=config.simulation_duration,
        rng=rng,
    )

    requests = create_requests(
        arrival_times=arrival_times,
        users=config.users,
        rng=rng,
    )

    architecture = Architecture(config, rng)

    result = SimulationResult(requests=requests)

    accepted_by_user: dict[int, int] = defaultdict(int)

    for request in requests:
        decision, operation_failed, node_id = architecture.decide(request)

        event = RequestEvent(
            request_id=request.request_id,
            user_id=request.user_id,
            arrival_time=request.arrival_time,
            decision_time=decision.decision_time,
            allowed=decision.allowed,
            distributed_operation_failed=operation_failed,
            node_id=node_id,
        )

        result.events.append(event)

        if decision.allowed:
            accepted_by_user[request.user_id] += 1

    result.accepted_by_user = dict(accepted_by_user)

    # The limiter owns the authoritative admitted timestamps.
    result.admitted_times = architecture.rate_limiter.admitted_times

    return result