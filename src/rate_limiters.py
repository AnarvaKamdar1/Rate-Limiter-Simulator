from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .generator import Request


@dataclass(frozen=True)
class Decision:
    """Result returned by a rate limiter."""

    allowed: bool
    decision_time: float


class SlidingWindowRateLimiter:
    """Exact rolling-window rate limiter."""

    def __init__(self, limit: int, window_size: float = 1.0) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        self.limit = limit
        self.window_size = window_size
        self._admitted_times: deque[float] = deque()
        self._all_admitted_times: list[float] = []

    def allow(
        self,
        request: Request,
        current_time: float,
    ) -> Decision:
        """Admit if fewer than limit requests are in the current rolling window."""
        del request

        # Remove timestamps that can no longer affect this request.
        # The deque is chronological, so this is O(number of expired entries).
        cutoff = current_time - self.window_size

        while self._admitted_times and self._admitted_times[0] <= cutoff:
            self._admitted_times.popleft()

        if len(self._admitted_times) < self.limit:
            self._admitted_times.append(current_time)
            self._all_admitted_times.append(current_time)
            return Decision(True, current_time)

        return Decision(False, current_time)

    @property
    def admitted_times(self) -> list[float]:
        """Return admitted timestamps for diagnostics."""
        return list(self._all_admitted_times)


class TokenBucketRateLimiter:
    """Continuous-time token bucket."""

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if refill_rate < 0:
            raise ValueError("refill_rate cannot be negative")

        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self.tokens = float(capacity)
        self.last_update_time = 0.0
        self._admitted_times: list[float] = []

    def _refill(self, current_time: float) -> None:
        elapsed = max(0.0, current_time - self.last_update_time)
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_update_time = max(self.last_update_time, current_time)

    def allow(
        self,
        request: Request,
        current_time: float,
    ) -> Decision:
        """Refill tokens to current simulated time, then consume one if available."""
        del request

        self._refill(current_time)

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            self._admitted_times.append(current_time)
            return Decision(True, current_time)

        return Decision(False, current_time)

    @property
    def admitted_times(self) -> list[float]:
        return list(self._admitted_times)


class LeakyBucketRateLimiter:
    """Queue-based leaky bucket with explicit service scheduling.

    ``decision_time`` is the scheduled service/start time.  Requests can wait
    in the bucket, so this value can be later than ``current_time`` and is
    therefore suitable for queueing-latency measurements.

    ``capacity`` is the maximum number of requests that are either currently
    being serviced or waiting for service.
    """

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if leak_rate <= 0:
            raise ValueError("leak_rate must be positive")

        self.capacity = capacity
        self.leak_rate = float(leak_rate)
        self._service_times: deque[float] = deque()
        self._admitted_times: list[float] = []
        self._service_interval = 1.0 / self.leak_rate

    def _remove_completed(self, current_time: float) -> None:
        # A request scheduled at t occupies the bucket until t + service
        # interval.  Do not remove it merely because its service has started.
        while (
            self._service_times
            and self._service_times[0] + self._service_interval <= current_time
        ):
            self._service_times.popleft()

    def allow(
        self,
        request: Request,
        current_time: float,
    ) -> Decision:
        """Admit the request if capacity is available and return its scheduled service time."""
        del request

        self._remove_completed(current_time)

        if len(self._service_times) >= self.capacity:
            return Decision(False, current_time)

        if self._service_times:
            service_time = max(
                current_time,
                self._service_times[-1] + self._service_interval,
            )
        else:
            service_time = current_time

        self._service_times.append(service_time)
        # Admission happens now; service may happen later.  Keep the admission
        # timestamp for rate-limit/overshoot analysis.
        self._admitted_times.append(current_time)

        return Decision(True, service_time)

    @property
    def queue_length(self) -> int:
        return len(self._service_times)

    @property
    def admitted_times(self) -> list[float]:
        return list(self._admitted_times)


def create_rate_limiter(
    algorithm: str,
    limit: int,
    window_size: float = 1.0,
    token_capacity: int | None = None,
    token_refill_rate: float | None = None,
    leaky_bucket_capacity: int | None = None,
    leaky_bucket_rate: float | None = None,
):
    """Create a configured rate limiter."""
    if algorithm == "sliding_window":
        return SlidingWindowRateLimiter(
            limit=limit,
            window_size=window_size,
        )

    if algorithm == "token_bucket":
        return TokenBucketRateLimiter(
            capacity=token_capacity if token_capacity is not None else limit,
            refill_rate=(
                token_refill_rate
                if token_refill_rate is not None
                else float(limit)
            ),
        )

    if algorithm == "leaky_bucket":
        return LeakyBucketRateLimiter(
            capacity=(
                leaky_bucket_capacity
                if leaky_bucket_capacity is not None
                else limit
            ),
            leak_rate=(
                leaky_bucket_rate
                if leaky_bucket_rate is not None
                else float(limit)
            ),
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")