"""Benchmark result contracts and artifact helpers."""

from LaughLM.benchmarks.result import (
    BenchmarkResult,
    CollectiveCount,
    CollectiveSummary,
    HBMMeasurement,
    KernelResolution,
    PhaseMeasurement,
    ThroughputMeasurement,
    TrialIdentity,
    canonical_result_json,
    result_sha256,
    write_benchmark_result,
)

__all__ = [
    "BenchmarkResult",
    "CollectiveCount",
    "CollectiveSummary",
    "HBMMeasurement",
    "KernelResolution",
    "PhaseMeasurement",
    "ThroughputMeasurement",
    "TrialIdentity",
    "canonical_result_json",
    "result_sha256",
    "write_benchmark_result",
]
