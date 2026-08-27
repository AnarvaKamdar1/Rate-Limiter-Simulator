from __future__ import annotations

import math

import numpy as np


def constant_arrivals(
    arrival_rate: float,
    duration: float,
    rng: np.random.Generator | None = None,
) -> list[float]:
    """Generate evenly spaced arrivals.

    The RNG argument is accepted so all traffic generators share one interface.
    Constant traffic itself is deterministic and therefore does not need randomness.
    """
    del rng

    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")

    interval = 1.0 / arrival_rate
    times: list[float] = []

    time = 0.0
    while time < duration:
        times.append(time)
        time += interval

    return times


def poisson_arrivals(
    arrival_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate Poisson-process arrivals using exponential inter-arrival times."""
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")

    times: list[float] = []
    time = 0.0

    while True:
        time += float(rng.exponential(scale=1.0 / arrival_rate))
        if time >= duration:
            break
        times.append(time)

    return times


def ramp_rate(
    time: float,
    arrival_rate: float,
    duration: float,
) -> float:
    """Return the instantaneous ramp rate.

    The rate starts at 25% and increases linearly to 100%.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    fraction = min(max(time / duration, 0.0), 1.0)
    return arrival_rate * (0.25 + 0.75 * fraction)


def _cumulative_ramp_intensity(
    time: float,
    arrival_rate: float,
    duration: float,
) -> float:
    """Integral of the ramp's instantaneous rate from zero to time."""
    start_rate = 0.25 * arrival_rate
    slope = 0.75 * arrival_rate / duration

    return start_rate * time + 0.5 * slope * time * time


def _solve_ramp_time(
    target_intensity: float,
    arrival_rate: float,
    duration: float,
) -> float:
    """Invert the cumulative ramp intensity equation."""
    start_rate = 0.25 * arrival_rate
    slope = 0.75 * arrival_rate / duration

    if slope == 0:
        return target_intensity / start_rate

    discriminant = start_rate * start_rate + 2.0 * slope * target_intensity

    return (-start_rate + math.sqrt(discriminant)) / slope


def ramp_arrivals(
    arrival_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate a linearly increasing non-homogeneous Poisson process.

    The instantaneous rate starts at 25% of arrival_rate and ends at
    arrival_rate. Exponential event mass is sampled and mapped through
    the integrated intensity function.
    """
    if arrival_rate <= 0:
        raise ValueError("arrival_rate must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")

    times: list[float] = []
    cumulative_target = 0.0

    while True:
        cumulative_target += float(rng.exponential(scale=1.0))

        time = _solve_ramp_time(
            cumulative_target,
            arrival_rate,
            duration,
        )

        if time >= duration:
            break

        times.append(time)

    return times


def generate_arrival_times(
    pattern: str,
    arrival_rate: float,
    duration: float,
    rng: np.random.Generator,
) -> list[float]:
    """Generate arrival timestamps for the selected traffic pattern."""
    if pattern == "constant":
        return constant_arrivals(arrival_rate, duration, rng)
    if pattern == "poisson":
        return poisson_arrivals(arrival_rate, duration, rng)
    if pattern == "ramp":
        return ramp_arrivals(arrival_rate, duration, rng)

    raise ValueError(f"Unsupported traffic pattern: {pattern}")