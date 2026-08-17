#!/usr/bin/env python3
"""Evaluate a saved LaughLM experiment candidate without runtime imports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _run_file(run_dir: Path, name: str) -> Path:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def _cache_state(manifest: dict[str, Any]) -> str:
    cache = manifest.get("compilation_cache", {})
    if not isinstance(cache, dict):
        return "unknown"
    if cache.get("cleared_before_run") is True:
        return "cold"
    count = cache.get("file_count_before_run")
    if not isinstance(count, int) or count < 0:
        return "unknown"
    return "warm" if count else "cold"


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if isinstance(row.get(field), (int, float))
        and math.isfinite(float(row[field]))
    ]


def _summary(rows: list[dict[str, Any]], skip_steps: int, last_n: int | None) -> dict[str, Any]:
    selected = rows[skip_steps:]
    if last_n is not None:
        if last_n <= 0:
            raise ValueError("--last-n must be positive")
        selected = selected[-last_n:]
    if not selected:
        raise ValueError("No metrics rows remain after selection")

    def statistic(field: str, reducer) -> float | None:
        values = _values(selected, field)
        return None if not values else float(reducer(values))

    memory_values = _values(
        selected,
        "device_memory_peak_bytes_in_use",
    )
    return {
        "rows_total": len(rows),
        "rows_selected": len(selected),
        "tokens_per_sec_median": statistic("tokens_per_sec", median),
        "device_tokens_per_sec_median": statistic("device_tokens_per_sec", median),
        "mfu_non_embedding_median": statistic("mfu_non_embedding", median),
        "loss_last": _values(selected, "loss")[-1] if _values(selected, "loss") else None,
        "total_step_time_mean": statistic("total_step_time", mean),
        "device_step_time_mean": statistic("device_step_time", mean),
        "device_memory_peak_bytes_max": max(memory_values) if memory_values else None,
    }


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("resolved_config", {})
    model = config.get("model", {}) if isinstance(config, dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    data = config.get("data", {}) if isinstance(config, dict) else {}
    manifest_data = manifest.get("data", {})
    if not isinstance(manifest_data, dict):
        manifest_data = {}
    jax_info = manifest.get("jax", {})
    if not isinstance(jax_info, dict):
        jax_info = {}
    cli_args = manifest.get("cli_args", {})
    if not isinstance(cli_args, dict):
        cli_args = {}
    identity = {
        key: model.get(key)
        for key in (
            "vocab_size",
            "d_model",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "max_seq_len",
        )
    } | {
        key: runtime.get(key)
        for key in (
            "seq_len",
            "micro_batch_per_device",
            "gradient_accumulation",
        )
    } | {
        "train_files": manifest_data.get("train_files") or data.get("train_files"),
        "validation_files": manifest_data.get("validation_files")
        or data.get("validation_files"),
        "token_dtype": manifest_data.get("token_dtype"),
        "hf_repo_id": cli_args.get("hf_repo_id") or data.get("hf_repo_id"),
        "hf_revision": cli_args.get("hf_revision") or data.get("hf_revision"),
        "train_shard_start": cli_args.get("train_shard_start")
        or data.get("train_shard_start"),
        "train_shard_count": cli_args.get("train_shard_count")
        or data.get("train_shard_count"),
        "validation_shard_start": cli_args.get("validation_shard_start")
        or data.get("validation_shard_start"),
        "validation_shard_count": cli_args.get("validation_shard_count")
        or data.get("validation_shard_count"),
        "process_count": jax_info.get("process_count"),
        "local_device_count": jax_info.get("local_device_count"),
        "devices": jax_info.get("devices"),
    }
    return identity


def evaluate_candidate(
    *,
    baseline_dir: Path,
    candidate_dir: Path,
    skip_steps: int,
    last_n: int | None,
    throughput_tolerance: float,
    max_loss_delta: float,
    memory_tolerance: float,
    require_memory: bool,
) -> dict[str, Any]:
    baseline_manifest = _load_json(_run_file(baseline_dir, "run_manifest.json"))
    candidate_manifest = _load_json(_run_file(candidate_dir, "run_manifest.json"))
    baseline = _summary(
        _load_rows(_run_file(baseline_dir, "metrics.jsonl")), skip_steps, last_n
    )
    candidate = _summary(
        _load_rows(_run_file(candidate_dir, "metrics.jsonl")), skip_steps, last_n
    )

    baseline_identity = _identity(baseline_manifest)
    candidate_identity = _identity(candidate_manifest)
    identity_mismatches = {
        key: {"baseline": baseline_identity[key], "candidate": candidate_identity[key]}
        for key in baseline_identity
        if baseline_identity[key] != candidate_identity[key]
    }
    baseline_tps = baseline["tokens_per_sec_median"]
    candidate_tps = candidate["tokens_per_sec_median"]
    throughput_delta = (
        None
        if baseline_tps in (None, 0) or candidate_tps is None
        else (candidate_tps / baseline_tps - 1.0) * 100.0
    )
    loss_delta = (
        None
        if baseline["loss_last"] is None or candidate["loss_last"] is None
        else candidate["loss_last"] - baseline["loss_last"]
    )
    baseline_memory = baseline["device_memory_peak_bytes_max"]
    candidate_memory = candidate["device_memory_peak_bytes_max"]
    memory_delta = (
        None
        if baseline_memory is None or candidate_memory is None
        else candidate_memory - baseline_memory
    )
    cache_baseline = _cache_state(baseline_manifest)
    cache_candidate = _cache_state(candidate_manifest)
    memory_ok = (
        baseline_memory is not None
        and candidate_memory is not None
        and candidate_memory <= baseline_memory * (1.0 + memory_tolerance)
    )
    checks = [
        {
            "name": "run identity",
            "passed": not identity_mismatches,
            "expected": "same model/data/batch comparison identity",
            "actual": identity_mismatches or "matched",
        },
        {
            "name": "candidate throughput",
            "passed": throughput_delta is not None
            and throughput_delta >= -throughput_tolerance,
            "expected": f">= baseline minus {throughput_tolerance:.2f}%",
            "actual": throughput_delta,
        },
        {
            "name": "candidate loss",
            "passed": loss_delta is not None and loss_delta <= max_loss_delta,
            "expected": f"loss delta <= {max_loss_delta:.6f}",
            "actual": loss_delta,
        },
        {
            "name": "memory evidence",
            "passed": memory_ok if require_memory else True,
            "expected": (
                f"candidate peak <= baseline + {memory_tolerance:.2%}"
                if require_memory
                else "optional"
            ),
            "actual": memory_delta,
        },
    ]
    return {
        "evaluation": "LaughLM experiment candidate",
        "status": "pass"
        if not identity_mismatches and all(check["passed"] for check in checks)
        else "fail",
        "baseline_dir": str(baseline_dir),
        "candidate_dir": str(candidate_dir),
        "selection": {"skip_steps": skip_steps, "last_n": last_n},
        "cache": {
            "baseline": cache_baseline,
            "candidate": cache_candidate,
            "compile_delta_eligible": cache_baseline == cache_candidate
            and cache_baseline != "unknown",
        },
        "baseline": baseline,
        "candidate": candidate,
        "deltas": {
            "throughput_percent": throughput_delta,
            "loss": loss_delta,
            "peak_memory_bytes": memory_delta,
        },
        "checks": checks,
        "note": "A passing static gate still requires TPU dispatch, fallback, and stability review.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved LaughLM experiment candidate."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--last-n", type=int)
    parser.add_argument("--throughput-tolerance", type=float, default=1.0)
    parser.add_argument("--max-loss-delta", type=float, default=0.05)
    parser.add_argument("--memory-tolerance", type=float, default=0.02)
    parser.add_argument("--require-memory", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate_candidate(
        baseline_dir=args.baseline.expanduser().resolve(),
        candidate_dir=args.candidate.expanduser().resolve(),
        skip_steps=args.skip_steps,
        last_n=args.last_n,
        throughput_tolerance=args.throughput_tolerance,
        max_loss_delta=args.max_loss_delta,
        memory_tolerance=args.memory_tolerance,
        require_memory=args.require_memory,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[candidate-evaluation] {report['status'].upper()}")
    print(f"[candidate-evaluation] report written: {output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
