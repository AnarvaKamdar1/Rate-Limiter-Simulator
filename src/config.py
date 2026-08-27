from __future__ import annotations

from dataclasses import dataclass, replace


# ---------------------------------------------------------------------------
# Supported values
# ---------------------------------------------------------------------------

TRAFFIC_PATTERNS = (
    "constant",
    "poisson",
    "ramp",
)

ALGORITHMS = (
    "sliding_window",
    "token_bucket",
    "leaky_bucket",
)

ARCHITECTURES = (
    "local",
    "distributed",
)


# ---------------------------------------------------------------------------
# Default manual simulation settings
# ---------------------------------------------------------------------------
#
# These values are used when running one simulation with run.py.
#
# They are deliberately scalar values because one SimulationConfig
# represents exactly one simulation.
# ---------------------------------------------------------------------------

DEFAULT_USERS = 50
DEFAULT_ARRIVAL_RATE = 100.0
DEFAULT_RATE_LIMIT = 50
DEFAULT_SIMULATION_DURATION = 10.0

DEFAULT_WINDOW_SIZE = 1.0

DEFAULT_TOKEN_CAPACITY = 50
DEFAULT_TOKEN_REFILL_RATE = 50.0

DEFAULT_LEAKY_BUCKET_CAPACITY = 50
DEFAULT_LEAKY_BUCKET_RATE = 50.0

DEFAULT_DISTRIBUTED_LATENCY_MS = 10.0
DEFAULT_FAILURE_RATE = 0.01
DEFAULT_FAIL_CLOSED = True
DEFAULT_DISTRIBUTED_NODES = 3


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
#
# These values are collections because the experiment suite tests every
# relevant combination.
#
# IMPORTANT:
#
# Distributed latency and failure rate are only experiment dimensions for
# the distributed architecture.
#
# For local architecture:
#   - distributed latency has no operational effect
#   - distributed failures have no operational effect
#
# Therefore local configurations are NOT duplicated for every latency and
# failure-rate combination.
#
# Local configurations:
#
#   2 users
# × 2 arrival rates
# × 3 traffic patterns
# × 3 algorithms
# = 36 configurations
#
# Distributed configurations:
#
#   2 users
# × 2 arrival rates
# × 3 traffic patterns
# × 3 algorithms
# × 2 distributed latencies
# × 2 failure rates
# = 144 configurations
#
# Total:
#
#   36 + 144 = 180 configurations
#
# ---------------------------------------------------------------------------

EXPERIMENT_USERS = (
    20,
    50,
)

EXPERIMENT_ARRIVAL_RATES = (
    50.0,
    100.0,
)

EXPERIMENT_TRAFFIC_PATTERNS = TRAFFIC_PATTERNS

EXPERIMENT_ALGORITHMS = ALGORITHMS

EXPERIMENT_ARCHITECTURES = ARCHITECTURES


# Distributed-only experiment dimensions.
#
# These must be collections, not aliases of the scalar manual defaults.
EXPERIMENT_DISTRIBUTED_LATENCY_MS = (
    1.0,
    10.0,
)

EXPERIMENT_FAILURE_RATE = (
    0.01,
    0.03,
)

# Fail-closed is currently fixed for every experiment.
EXPERIMENT_FAIL_CLOSED = True


# Number of Monte Carlo repetitions for each configuration.
#
# 180 configurations × 10 runs = 1,800 simulations.
#
# Increase this to 25, 30, or 50 for a larger final experiment.
DEFAULT_MONTE_CARLO_RUNS = 50


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConfig:
    """All inputs required to run one simulation."""

    # Traffic configuration.
    users: int = DEFAULT_USERS
    arrival_rate: float = DEFAULT_ARRIVAL_RATE
    traffic_pattern: str = "constant"

    # Rate-limiting configuration.
    algorithm: str = "sliding_window"

    # System architecture.
    architecture: str = "local"

    # Distributed-system behavior.
    #
    # These are scalar values because a SimulationConfig represents exactly
    # one simulation. The experiment generator chooses one value from the
    # experiment dimensions above.
    distributed_latency_ms: float = DEFAULT_DISTRIBUTED_LATENCY_MS
    failure_rate: float = DEFAULT_FAILURE_RATE
    fail_closed: bool = DEFAULT_FAIL_CLOSED
    distributed_nodes: int = DEFAULT_DISTRIBUTED_NODES

    # Main rate limit.
    rate_limit: int = DEFAULT_RATE_LIMIT
    simulation_duration: float = DEFAULT_SIMULATION_DURATION

    # Sliding-window settings.
    window_size: float = DEFAULT_WINDOW_SIZE

    # Token-bucket settings.
    token_capacity: int = DEFAULT_TOKEN_CAPACITY
    token_refill_rate: float = DEFAULT_TOKEN_REFILL_RATE

    # Leaky-bucket settings.
    leaky_bucket_capacity: int = DEFAULT_LEAKY_BUCKET_CAPACITY
    leaky_bucket_rate: float = DEFAULT_LEAKY_BUCKET_RATE

    # Random seed.
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the configuration when it is created."""

        if self.users <= 0:
            raise ValueError(
                "users must be positive"
            )

        if self.arrival_rate <= 0:
            raise ValueError(
                "arrival_rate must be positive"
            )

        if self.rate_limit <= 0:
            raise ValueError(
                "rate_limit must be positive"
            )

        if self.simulation_duration <= 0:
            raise ValueError(
                "simulation_duration must be positive"
            )

        if self.traffic_pattern not in TRAFFIC_PATTERNS:
            raise ValueError(
                f"Unsupported traffic pattern: "
                f"{self.traffic_pattern}"
            )

        if self.algorithm not in ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm: "
                f"{self.algorithm}"
            )

        if self.architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: "
                f"{self.architecture}"
            )

        if self.distributed_latency_ms < 0:
            raise ValueError(
                "distributed_latency_ms cannot be negative"
            )

        if not 0 <= self.failure_rate <= 1:
            raise ValueError(
                "failure_rate must be between 0 and 1"
            )

        if self.distributed_nodes <= 0:
            raise ValueError(
                "distributed_nodes must be positive"
            )

        if self.window_size <= 0:
            raise ValueError(
                "window_size must be positive"
            )

        if self.token_capacity <= 0:
            raise ValueError(
                "token_capacity must be positive"
            )

        if self.token_refill_rate < 0:
            raise ValueError(
                "token_refill_rate cannot be negative"
            )

        if self.leaky_bucket_capacity <= 0:
            raise ValueError(
                "leaky_bucket_capacity must be positive"
            )

        if self.leaky_bucket_rate <= 0:
            raise ValueError(
                "leaky_bucket_rate must be positive"
            )

    def with_seed(self, seed: int) -> "SimulationConfig":
        """Return the same configuration with a different random seed."""
        return replace(
            self,
            seed=seed,
        )

    def csv_dict(self) -> dict[str, object]:
        """
        Return the configuration fields that should be written to CSV.

        Internal settings such as token capacity and bucket rates are
        intentionally omitted because the initial experiment matrix only
        treats rate_limit as the configurable rate-limit dimension.
        """
        return {
            "users": self.users,
            "arrival_rate": self.arrival_rate,
            "traffic_pattern": self.traffic_pattern,
            "algorithm": self.algorithm,
            "architecture": self.architecture,
            "distributed_latency": self.distributed_latency_ms,
            "failure_rate": self.failure_rate,
            "fail_closed": self.fail_closed,
            "rate_limit": self.rate_limit,
            "simulation_duration": self.simulation_duration,
        }