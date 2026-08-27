from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Request:
    """A single simulated request."""

    request_id: int
    user_id: int
    arrival_time: float
    metadata: dict[str, object] = field(default_factory=dict)


def create_requests(
    arrival_times: Iterable[float],
    users: int,
    rng: np.random.Generator,
) -> list[Request]:
    """Create requests from arrival timestamps and assign users reproducibly."""
    if users <= 0:
        raise ValueError("users must be positive")

    requests: list[Request] = []

    for request_id, arrival_time in enumerate(arrival_times):
        user_id = int(rng.integers(0, users))
        requests.append(
            Request(
                request_id=request_id,
                user_id=user_id,
                arrival_time=float(arrival_time),
            )
        )

    return requests


def generate_requests(
    arrival_times: Iterable[float],
    users: int,
    rng: np.random.Generator,
) -> list[Request]:
    """Compatibility-friendly alias for create_requests."""
    return create_requests(arrival_times, users, rng)