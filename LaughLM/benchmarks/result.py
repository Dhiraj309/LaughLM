"""Immutable, JSON-serializable benchmark result records.

The record is deliberately independent of JAX and the benchmark runner.  A
runner can collect device-specific details without making the result contract
depend on a particular runtime or kernel package.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RESULT_SCHEMA = "laughlm_benchmark_result_v1"
RESULT_VERSION = 1
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_nonnegative(value: int | float, field: str) -> None:
    if value < 0:
        raise ValueError(f"{field} must be non-negative")


def _require_digest(value: str, field: str) -> None:
    _require_nonempty(value, field)
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class TrialIdentity:
    """Stable identity for one benchmark repetition."""

    benchmark_id: str
    trial_id: str
    repetition: int
    seed: int

    def __post_init__(self) -> None:
        _require_nonempty(self.benchmark_id, "trial.benchmark_id")
        _require_nonempty(self.trial_id, "trial.trial_id")
        _require_nonnegative(self.repetition, "trial.repetition")
        _require_nonnegative(self.seed, "trial.seed")


@dataclass(frozen=True)
class PhaseMeasurement:
    """Step, duration, and token counts for warm-up or measured work."""

    steps: int
    duration_s: float
    tokens: int

    def __post_init__(self) -> None:
        _require_nonnegative(self.steps, "phase.steps")
        _require_nonnegative(self.duration_s, "phase.duration_s")
        _require_nonnegative(self.tokens, "phase.tokens")


@dataclass(frozen=True)
class ThroughputMeasurement:
    """Derived rates for the measured phase."""

    tokens_per_second: float
    steps_per_second: float

    def __post_init__(self) -> None:
        _require_nonnegative(
            self.tokens_per_second, "throughput.tokens_per_second"
        )
        _require_nonnegative(self.steps_per_second, "throughput.steps_per_second")


@dataclass(frozen=True)
class HBMMeasurement:
    """Backend-reported HBM counters; unavailable counters remain ``None``."""

    peak_bytes: int | None = None
    allocated_bytes: int | None = None
    limit_bytes: int | None = None

    def __post_init__(self) -> None:
        for field in ("peak_bytes", "allocated_bytes", "limit_bytes"):
            value = getattr(self, field)
            if value is not None:
                _require_nonnegative(value, f"hbm.{field}")


@dataclass(frozen=True)
class CollectiveCount:
    """Count for one named collective operation."""

    name: str
    count: int

    def __post_init__(self) -> None:
        _require_nonempty(self.name, "collective.name")
        _require_nonnegative(self.count, "collective.count")


@dataclass(frozen=True)
class CollectiveSummary:
    """Compiler/profile-derived collective counts."""

    operations: tuple[CollectiveCount, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.operations, tuple):
            raise TypeError("collectives.operations must be a tuple")


@dataclass(frozen=True)
class KernelResolution:
    """Requested implementation and the implementation actually executed."""

    operation: str
    requested: str
    resolved: str

    def __post_init__(self) -> None:
        _require_nonempty(self.operation, "kernel.operation")
        _require_nonempty(self.requested, "kernel.requested")
        _require_nonempty(self.resolved, "kernel.resolved")


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete immutable result envelope for one benchmark trial."""

    trial: TrialIdentity
    config_digest: str
    environment_digest: str
    compile_time_s: float | None
    warmup: PhaseMeasurement
    measured: PhaseMeasurement
    throughput: ThroughputMeasurement
    hbm: HBMMeasurement
    collectives: CollectiveSummary
    kernels: tuple[KernelResolution, ...]

    def __post_init__(self) -> None:
        _require_digest(self.config_digest, "config_digest")
        _require_digest(self.environment_digest, "environment_digest")
        if self.compile_time_s is not None:
            _require_nonnegative(self.compile_time_s, "compile_time_s")
        if not isinstance(self.kernels, tuple):
            raise TypeError("kernels must be a tuple")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON shape used by benchmark tools."""

        return {
            "result_schema": RESULT_SCHEMA,
            "result_version": RESULT_VERSION,
            "trial": {
                "benchmark_id": self.trial.benchmark_id,
                "trial_id": self.trial.trial_id,
                "repetition": self.trial.repetition,
                "seed": self.trial.seed,
            },
            "config_digest": self.config_digest,
            "environment_digest": self.environment_digest,
            "compile_time_s": self.compile_time_s,
            "warmup": _phase_to_dict(self.warmup),
            "measured": _phase_to_dict(self.measured),
            "throughput": {
                "tokens_per_second": self.throughput.tokens_per_second,
                "steps_per_second": self.throughput.steps_per_second,
            },
            "hbm": {
                "peak_bytes": self.hbm.peak_bytes,
                "allocated_bytes": self.hbm.allocated_bytes,
                "limit_bytes": self.hbm.limit_bytes,
            },
            "collectives": {
                "operations": [
                    {"name": item.name, "count": item.count}
                    for item in self.collectives.operations
                ]
            },
            "kernels": [
                {
                    "operation": item.operation,
                    "requested": item.requested,
                    "resolved": item.resolved,
                }
                for item in self.kernels
            ],
        }


def _phase_to_dict(phase: PhaseMeasurement) -> dict[str, int | float]:
    return {
        "steps": phase.steps,
        "duration_s": phase.duration_s,
        "tokens": phase.tokens,
    }


def canonical_result_json(result: BenchmarkResult) -> str:
    """Serialize a result deterministically for hashing and comparison."""

    return json.dumps(
        result.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def result_sha256(result: BenchmarkResult) -> str:
    """Return the SHA-256 digest of the canonical result envelope."""

    return hashlib.sha256(canonical_result_json(result).encode("utf-8")).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _production_checkpoint_root() -> Path:
    return (Path.cwd() / "checkpoints" / "production").resolve()


def _reject_unsafe_output(output: Path) -> None:
    production = _production_checkpoint_root()
    parts = tuple(part.casefold() for part in output.parts)
    has_production_component = any(
        parts[index : index + 2] == ("checkpoints", "production")
        for index in range(len(parts) - 1)
    )
    if (
        has_production_component
        or _is_within(output, production)
        or _is_within(production, output)
    ):
        raise ValueError(
            "Benchmark output must not overlap checkpoints/production: "
            f"{output}"
        )


def write_benchmark_result(
    result: BenchmarkResult,
    output_dir: str | Path,
) -> Path:
    """Write ``result.json`` to a new, non-production output directory.

    Existing directories are rejected even when they are empty, preventing a
    repeated trial from silently replacing evidence from an earlier run.
    """

    output = Path(output_dir).expanduser().resolve()
    _reject_unsafe_output(output)
    if output.exists():
        raise FileExistsError(
            f"Benchmark output directory already exists: {output}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    result_path = output / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            result.to_dict(),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return result_path
