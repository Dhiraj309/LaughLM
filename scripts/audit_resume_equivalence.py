#!/usr/bin/env python3
"""Compare uninterrupted and resumed run artifacts without runtime imports."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _rows(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("step"), int):
                result[int(row["step"])] = row
    return result


def _metadata(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def audit_resume_equivalence(
    *,
    uninterrupted_metrics: Path,
    resumed_metrics: Path,
    split_step: int,
    loss_tolerance: float,
    uninterrupted_metadata: Path | None = None,
    resumed_metadata: Path | None = None,
) -> dict[str, Any]:
    baseline = _rows(uninterrupted_metrics)
    resumed = _rows(resumed_metrics)
    common_steps = sorted(step for step in resumed if step > split_step and step in baseline)
    mismatches: list[dict[str, Any]] = []
    for step in common_steps:
        left, right = baseline[step], resumed[step]
        for field in ("loss", "tokens_processed"):
            expected, actual = left.get(field), right.get(field)
            if not isinstance(expected, (int, float)) or not isinstance(actual, (int, float)):
                mismatches.append({"step": step, "field": field, "expected": expected, "actual": actual})
            elif not math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=loss_tolerance):
                mismatches.append({"step": step, "field": field, "expected": expected, "actual": actual})

    metadata_check = {"checked": False, "passed": True}
    left_meta, right_meta = _metadata(uninterrupted_metadata), _metadata(resumed_metadata)
    if left_meta is not None or right_meta is not None:
        left_next = left_meta.get("data_iterator", {}).get("next_batch_index") if left_meta else None
        right_next = right_meta.get("data_iterator", {}).get("next_batch_index") if right_meta else None
        metadata_check = {"checked": True, "passed": isinstance(left_next, int) and isinstance(right_next, int) and right_next >= left_next, "uninterrupted_next_batch_index": left_next, "resumed_next_batch_index": right_next}

    passed = bool(common_steps) and not mismatches and metadata_check["passed"]
    return {
        "audit": "LaughLM resume equivalence",
        "status": "pass" if passed else "fail",
        "split_step": int(split_step),
        "common_steps_checked": common_steps,
        "mismatches": mismatches[:50],
        "metadata": metadata_check,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit resume equivalence from saved metrics.")
    parser.add_argument("--uninterrupted-metrics", type=Path, required=True)
    parser.add_argument("--resumed-metrics", type=Path, required=True)
    parser.add_argument("--split-step", type=int, required=True)
    parser.add_argument("--loss-tolerance", type=float, default=1e-6)
    parser.add_argument("--uninterrupted-metadata", type=Path)
    parser.add_argument("--resumed-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_resume_equivalence(
        uninterrupted_metrics=args.uninterrupted_metrics.expanduser().resolve(),
        resumed_metrics=args.resumed_metrics.expanduser().resolve(),
        split_step=args.split_step,
        loss_tolerance=args.loss_tolerance,
        uninterrupted_metadata=args.uninterrupted_metadata.expanduser().resolve() if args.uninterrupted_metadata else None,
        resumed_metadata=args.resumed_metadata.expanduser().resolve() if args.resumed_metadata else None,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[resume-audit] report written: {output}")
    print(f"[resume-audit] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
