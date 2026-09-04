"""CPU tests for the immutable Gen-2 benchmark result contract."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

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


def _result() -> BenchmarkResult:
    return BenchmarkResult(
        trial=TrialIdentity(
            benchmark_id="llama-dense-tiny",
            trial_id="trial-0001",
            repetition=1,
            seed=17,
        ),
        config_digest="a" * 64,
        environment_digest="b" * 64,
        compile_time_s=1.25,
        warmup=PhaseMeasurement(steps=2, duration_s=0.5, tokens=128),
        measured=PhaseMeasurement(steps=4, duration_s=1.0, tokens=256),
        throughput=ThroughputMeasurement(
            tokens_per_second=256.0,
            steps_per_second=4.0,
        ),
        hbm=HBMMeasurement(
            peak_bytes=1024,
            allocated_bytes=768,
            limit_bytes=2048,
        ),
        collectives=CollectiveSummary(
            operations=(CollectiveCount("all-reduce", 3),)
        ),
        kernels=(
            KernelResolution(
                operation="attention",
                requested="native",
                resolved="native",
            ),
        ),
    )


def test_result_is_immutable_and_has_required_sections() -> None:
    result = _result()

    with pytest.raises(FrozenInstanceError):
        result.compile_time_s = 2.0

    payload = result.to_dict()
    assert payload["result_schema"] == "laughlm_benchmark_result_v1"
    assert payload["config_digest"] == "a" * 64
    assert payload["environment_digest"] == "b" * 64
    assert payload["warmup"]["steps"] == 2
    assert payload["measured"]["tokens"] == 256
    assert payload["hbm"]["peak_bytes"] == 1024
    assert payload["collectives"]["operations"] == [
        {"name": "all-reduce", "count": 3}
    ]
    assert payload["kernels"][0]["resolved"] == "native"


def test_canonical_serialization_and_digest_are_stable() -> None:
    result = _result()

    first = canonical_result_json(result)
    second = canonical_result_json(result)

    assert first == second
    assert len(result_sha256(result)) == 64
    assert result_sha256(result) == result_sha256(result)


def test_result_writes_only_to_a_new_directory(tmp_path: Path) -> None:
    result_path = write_benchmark_result(result=_result(), output_dir=tmp_path / "trial")

    assert result_path.name == "result.json"
    assert result_path.is_file()
    assert '"result_schema": "laughlm_benchmark_result_v1"' in result_path.read_text()

    with pytest.raises(FileExistsError, match="already exists"):
        write_benchmark_result(result=_result(), output_dir=tmp_path / "trial")


def test_result_rejects_production_checkpoint_overlap(tmp_path: Path) -> None:
    production = tmp_path / "checkpoints" / "production" / "trial"
    with pytest.raises(ValueError, match="checkpoints/production"):
        write_benchmark_result(result=_result(), output_dir=production)


def test_result_rejects_invalid_digest() -> None:
    with pytest.raises(ValueError, match="config_digest"):
        BenchmarkResult(
            trial=_result().trial,
            config_digest="not-a-digest",
            environment_digest="b" * 64,
            compile_time_s=None,
            warmup=_result().warmup,
            measured=_result().measured,
            throughput=_result().throughput,
            hbm=_result().hbm,
            collectives=_result().collectives,
            kernels=_result().kernels,
        )
