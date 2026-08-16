#!/usr/bin/env python3
"""Audit saved LaughLM checkpoint artifacts without importing runtime code.

The audit reads Orbax step-directory names and LaughLM JSON sidecars. It does
not restore a checkpoint, import JAX/Orbax, or inspect model tensors.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_PATTERN = re.compile(r"^(?:step_)?(\d+)$")
METADATA_PATTERN = re.compile(r"^step_(\d+)\.json$")


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


def _step_from_name(name: str) -> int | None:
    match = STEP_PATTERN.fullmatch(name)
    return int(match.group(1)) if match else None


def _metadata_step(path: Path) -> int | None:
    match = METADATA_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def audit_checkpoint_dir(
    *,
    checkpoint_dir: Path,
    expected_max_to_keep: int | None,
    require_run_manifest: bool,
) -> dict[str, Any]:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    if not checkpoint_dir.is_dir() or checkpoint_dir.is_symlink():
        raise FileNotFoundError(f"Checkpoint directory is invalid: {checkpoint_dir}")

    checks: list[dict[str, Any]] = []
    step_dirs: dict[int, Path] = {}
    invalid_step_entries: list[str] = []
    for entry in sorted(checkpoint_dir.iterdir()):
        if entry.name in {"checkpoint_metadata", "run_manifest.json", "metrics.jsonl"}:
            continue
        step = _step_from_name(entry.name)
        if step is None:
            continue
        if entry.is_symlink() or not entry.is_dir():
            invalid_step_entries.append(entry.name)
            continue
        if step in step_dirs:
            invalid_step_entries.append(entry.name)
            continue
        step_dirs[step] = entry

    steps = sorted(step_dirs)
    _record(
        checks,
        "Orbax checkpoint steps",
        passed=bool(steps) and not invalid_step_entries,
        expected="at least one numeric checkpoint step directory",
        actual={"steps": steps, "invalid_entries": invalid_step_entries},
    )

    metadata_dir = checkpoint_dir / "checkpoint_metadata"
    metadata_paths: list[Path] = []
    metadata_issues: list[str] = []
    if metadata_dir.is_symlink():
        metadata_issues.append("checkpoint_metadata is a symlink")
    elif metadata_dir.is_dir():
        for entry in sorted(metadata_dir.iterdir()):
            if entry.is_symlink() or not entry.is_file():
                metadata_issues.append(entry.name)
                continue
            if _metadata_step(entry) is None:
                metadata_issues.append(entry.name)
                continue
            metadata_paths.append(entry)

    metadata_steps = sorted(
        step for step in (_metadata_step(path) for path in metadata_paths)
        if step is not None
    )
    _record(
        checks,
        "checkpoint metadata sidecars",
        passed=bool(metadata_steps) and not metadata_issues,
        expected="step_*.json sidecars with no unexpected entries",
        actual={"steps": metadata_steps, "issues": metadata_issues},
    )

    _record(
        checks,
        "checkpoint/metadata step alignment",
        passed=bool(steps) and steps == metadata_steps,
        expected="Orbax steps exactly match metadata sidecar steps",
        actual={"checkpoint_steps": steps, "metadata_steps": metadata_steps},
    )

    metadata_value_issues: list[str] = []
    metadata_formats: dict[str, int] = {}
    for path in metadata_paths:
        step = _metadata_step(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            metadata_value_issues.append(f"{path.name}: {type(exc).__name__}")
            continue
        if not isinstance(value, dict):
            metadata_value_issues.append(f"{path.name}: expected JSON object")
            continue
        if value.get("step") != step:
            metadata_value_issues.append(
                f"{path.name}: internal step={value.get('step')!r}, filename={step}"
            )
        metadata_formats[str(value.get("format"))] = (
            metadata_formats.get(str(value.get("format")), 0) + 1
        )
    _record(
        checks,
        "metadata contents",
        passed=bool(metadata_paths) and not metadata_value_issues,
        expected="valid JSON with internal step matching filename",
        actual={"issues": metadata_value_issues, "formats": metadata_formats},
    )

    if expected_max_to_keep is not None:
        retention_passed = (
            expected_max_to_keep > 0 and len(steps) <= expected_max_to_keep
        )
        _record(
            checks,
            "checkpoint retention",
            passed=retention_passed,
            expected=f"at most {expected_max_to_keep} retained checkpoint(s)",
            actual=len(steps),
        )

    run_manifest_path = checkpoint_dir / "run_manifest.json"
    metrics_path = checkpoint_dir / "metrics.jsonl"
    provenance_passed = run_manifest_path.is_file() and metrics_path.is_file()
    if require_run_manifest:
        _record(
            checks,
            "run provenance artifacts",
            passed=provenance_passed,
            expected="run_manifest.json and metrics.jsonl",
            actual={
                "run_manifest": run_manifest_path.is_file(),
                "metrics": metrics_path.is_file(),
            },
        )

    latest_step = max(steps) if steps else None
    latest_metadata = max(metadata_steps) if metadata_steps else None
    _record(
        checks,
        "latest checkpoint alignment",
        passed=latest_step is not None and latest_step == latest_metadata,
        expected="latest checkpoint has matching latest metadata",
        actual={"checkpoint": latest_step, "metadata": latest_metadata},
    )

    passed = all(check["passed"] for check in checks)
    return {
        "audit": "LaughLM checkpoint artifacts",
        "status": "pass" if passed else "fail",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "expected_max_to_keep": expected_max_to_keep,
        "require_run_manifest": require_run_manifest,
        "steps": steps,
        "metadata_steps": metadata_steps,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaughLM checkpoint artifacts without runtime imports."
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--expected-max-to-keep", type=int)
    parser.add_argument("--require-run-manifest", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_checkpoint_dir(
        checkpoint_dir=args.checkpoint_dir,
        expected_max_to_keep=args.expected_max_to_keep,
        require_run_manifest=args.require_run_manifest,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[checkpoint-audit] report written: {output}")
    print(f"[checkpoint-audit] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
