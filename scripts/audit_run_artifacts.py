#!/usr/bin/env python3
"""Audit LaughLM run manifests and metrics without runtime imports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CORE_PACKAGES = ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint")
TOKEN_STORAGE = {
    "uint16": 2,
    "uint64": 8,
}
REQUIRED_TIMING_FIELDS = (
    "total_step_time",
    "data_wait_time",
    "host_batch_prepare_time",
    "device_put_time",
    "device_step_time",
)
REQUIRED_METRIC_FIELDS = (
    "step",
    "loss",
    "tokens_per_sec",
    "device_tokens_per_sec",
    "mfu_non_embedding",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc.msg}")
                continue
            if not isinstance(value, dict):
                errors.append(f"line {line_number}: expected JSON object")
                continue
            rows.append(value)
    return rows, errors


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _record(
    checks: list[dict[str, Any]],
    name: str,
    *,
    passed: bool,
    expected: Any = None,
    actual: Any = None,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
        }
    )


def audit_run(
    *,
    manifest_path: Path,
    metrics_path: Path,
    require_checkpoint_timings: bool,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    rows, jsonl_errors = _load_jsonl(metrics_path)
    checks: list[dict[str, Any]] = []

    manifest_version = manifest.get("manifest_version")
    _record(
        checks,
        "manifest version",
        passed=isinstance(manifest_version, int) and manifest_version >= 2,
        expected="manifest_version >= 2",
        actual=manifest_version,
    )
    _record(
        checks,
        "git revision",
        passed=bool(manifest.get("git_revision")),
        expected="non-empty git revision",
        actual=manifest.get("git_revision"),
    )

    package_versions = manifest.get("package_versions")
    _record(
        checks,
        "dependency versions",
        passed=isinstance(package_versions, dict)
        and all(package_versions.get(name) for name in CORE_PACKAGES),
        expected="core runtime package versions",
        actual=package_versions,
    )

    resolved_config = manifest.get("resolved_config")
    _record(
        checks,
        "resolved configuration",
        passed=isinstance(resolved_config, dict)
        and all(
            isinstance(resolved_config.get(section), dict)
            for section in ("model", "architecture", "runtime", "loss", "spmd")
        ),
        expected="model, architecture, runtime, loss, and spmd sections",
        actual=sorted(resolved_config) if isinstance(resolved_config, dict) else None,
    )

    attention_contract = manifest.get("attention_contract")
    _record(
        checks,
        "attention contract",
        passed=isinstance(attention_contract, dict)
        and all(
            attention_contract.get(key) is not None
            for key in ("variant", "implementation_requested", "fallback_policy", "num_heads")
        ),
        expected="attention variant, implementation, fallback, and head geometry",
        actual=attention_contract,
    )
    loss_contract = manifest.get("loss_contract")
    _record(
        checks,
        "loss contract",
        passed=isinstance(loss_contract, dict)
        and all(
            loss_contract.get(key) is not None
            for key in ("loss_backend_requested", "fallback_policy", "lm_head_layout")
        ),
        expected="loss backend, fallback policy, and LM-head layout",
        actual=loss_contract,
    )

    data_contract = manifest.get("data")
    token_dtype = (
        data_contract.get("token_dtype")
        if isinstance(data_contract, dict)
        else None
    )
    token_itemsize = (
        data_contract.get("token_itemsize_bytes")
        if isinstance(data_contract, dict)
        else None
    )
    expected_itemsize = TOKEN_STORAGE.get(token_dtype)
    _record(
        checks,
        "token storage contract",
        passed=(
            isinstance(data_contract, dict)
            and expected_itemsize is not None
            and token_itemsize == expected_itemsize
        ),
        expected="token_dtype is uint16/uint64 with matching itemsize",
        actual={"dtype": token_dtype, "itemsize_bytes": token_itemsize},
    )

    shard_errors: list[str] = []
    for split in ("train", "validation"):
        details = (
            data_contract.get(f"{split}_shard_details")
            if isinstance(data_contract, dict)
            else None
        )
        if not isinstance(details, list) or not details:
            shard_errors.append(f"{split}: missing shard details")
            continue
        for index, detail in enumerate(details):
            prefix = f"{split}[{index}]"
            if not isinstance(detail, dict):
                shard_errors.append(f"{prefix}: expected object")
                continue
            if not detail.get("path") or not detail.get("filename"):
                shard_errors.append(f"{prefix}: missing path or filename")
            if detail.get("dtype") != token_dtype:
                shard_errors.append(f"{prefix}: dtype does not match manifest")
            if detail.get("itemsize_bytes") != token_itemsize:
                shard_errors.append(f"{prefix}: itemsize does not match manifest")
            if detail.get("size_aligned") is not True:
                shard_errors.append(f"{prefix}: shard size is not aligned")
            byte_size = detail.get("byte_size")
            token_count = detail.get("token_count")
            if (
                not isinstance(byte_size, int)
                or isinstance(byte_size, bool)
                or byte_size <= 0
            ):
                shard_errors.append(f"{prefix}: invalid byte_size")
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count <= 0
            ):
                shard_errors.append(f"{prefix}: invalid token_count")
            if (
                isinstance(byte_size, int)
                and not isinstance(byte_size, bool)
                and isinstance(token_itemsize, int)
                and byte_size % token_itemsize != 0
            ):
                shard_errors.append(f"{prefix}: byte_size/itemsize mismatch")
            if (
                isinstance(byte_size, int)
                and not isinstance(byte_size, bool)
                and isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and isinstance(token_itemsize, int)
                and byte_size != token_count * token_itemsize
            ):
                shard_errors.append(f"{prefix}: token_count does not match byte_size")
    _record(
        checks,
        "shard manifest integrity",
        passed=not shard_errors,
        expected="non-empty aligned train/validation shard details with consistent counts",
        actual=shard_errors[:20] or "valid",
    )

    cache = manifest.get("compilation_cache")
    _record(
        checks,
        "compilation cache contract",
        passed=isinstance(cache, dict)
        and all(
            key in cache
            for key in (
                "configured",
                "directory",
                "file_count_before_run",
                "cleared_before_run",
            )
        ),
        expected="cache directory and explicit cold/warm state",
        actual=cache,
    )

    _record(
        checks,
        "metrics JSONL parse",
        passed=bool(rows) and not jsonl_errors,
        expected="at least one valid metrics row and no JSON errors",
        actual={"rows": len(rows), "errors": jsonl_errors},
    )
    for field in REQUIRED_METRIC_FIELDS:
        missing = sum(field not in row for row in rows)
        _record(
            checks,
            f"metrics field: {field}",
            passed=bool(rows) and missing == 0,
            expected="present in every metrics row",
            actual={"missing_rows": missing, "rows": len(rows)},
        )
    for field in REQUIRED_TIMING_FIELDS:
        missing = sum(field not in row for row in rows)
        _record(
            checks,
            f"timing field: {field}",
            passed=bool(rows) and missing == 0,
            expected="present in every metrics row",
            actual={"missing_rows": missing, "rows": len(rows)},
        )

    non_finite: list[str] = []
    for row_index, row in enumerate(rows):
        for field in REQUIRED_METRIC_FIELDS[1:] + REQUIRED_TIMING_FIELDS:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                non_finite.append(f"row {row_index} field {field}")
    _record(
        checks,
        "metrics numeric values",
        passed=not non_finite,
        expected="finite numeric metric/timing values",
        actual=non_finite[:20] or "finite",
    )

    compile_evidence = any(
        row.get("first_step_compile_plus_execute_time") is not None
        for row in rows
    )
    _record(
        checks,
        "first-step compile evidence",
        passed=compile_evidence,
        expected="first_step_compile_plus_execute_time recorded",
        actual="present" if compile_evidence else "missing",
    )

    checkpoint_timings_path = metrics_path.parent / "checkpoint_timings.jsonl"
    checkpoint_timings_present = checkpoint_timings_path.is_file()
    _record(
        checks,
        "checkpoint timing artifact",
        passed=checkpoint_timings_present if require_checkpoint_timings else True,
        expected="checkpoint_timings.jsonl when required",
        actual=str(checkpoint_timings_path) if checkpoint_timings_present else "missing",
    )

    passed = all(check["passed"] for check in checks)
    return {
        "audit": "LaughLM run artifacts",
        "status": "pass" if passed else "fail",
        "manifest_path": str(manifest_path),
        "metrics_path": str(metrics_path),
        "rows": len(rows),
        "require_checkpoint_timings": require_checkpoint_timings,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaughLM run artifacts without runtime imports."
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--require-checkpoint-timings", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.run_dir is not None and (args.manifest is not None or args.metrics is not None):
        parser.error("use --run-dir or explicit --manifest/--metrics, not both")
    if args.run_dir is None and (args.manifest is None or args.metrics is None):
        parser.error("provide --run-dir or both --manifest and --metrics")

    if args.run_dir is not None:
        manifest_path = args.run_dir / "run_manifest.json"
        metrics_path = args.run_dir / "metrics.jsonl"
    else:
        manifest_path = args.manifest
        metrics_path = args.metrics

    manifest_path = manifest_path.expanduser().resolve()
    metrics_path = metrics_path.expanduser().resolve()
    report = audit_run(
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        require_checkpoint_timings=args.require_checkpoint_timings,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[run-audit] report written: {output}")
    print(f"[run-audit] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
