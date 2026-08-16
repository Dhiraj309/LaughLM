#!/usr/bin/env python3
"""Aggregate saved LaughLM audit reports into one release-readiness result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_report(path: Path, label: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {label} report: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object for {label}: {resolved}")
    return value


def _result(label: str, path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": label,
        "path": str(path.expanduser().resolve()),
        "status": report.get("status", "missing"),
        "passed": report.get("status") == "pass",
    }


def audit_readiness(
    *,
    checkpoint_audit: Path,
    run_audit: Path,
    release_audit: Path,
    bundle_verification: Path,
    candidate_reports: list[Path],
) -> dict[str, Any]:
    required = [
        ("checkpoint_artifacts", checkpoint_audit),
        ("run_artifacts", run_audit),
        ("release_contract", release_audit),
        ("bundle_verification", bundle_verification),
    ]
    results = []
    for label, path in required:
        results.append(_result(label, path, _load_report(path, label)))
    for index, path in enumerate(candidate_reports, start=1):
        label = f"candidate_{index}"
        results.append(_result(label, path, _load_report(path, label)))

    passed = all(result["passed"] for result in results)
    return {
        "audit": "LaughLM release readiness",
        "status": "pass" if passed else "fail",
        "required_reports": len(required),
        "candidate_reports": len(candidate_reports),
        "reports": results,
        "next_gate": (
            "TPU validation and operational handoff"
            if passed
            else "resolve failing or missing audit reports"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate LaughLM release audit reports without runtime imports."
    )
    parser.add_argument("--checkpoint-audit", required=True, type=Path)
    parser.add_argument("--run-audit", required=True, type=Path)
    parser.add_argument("--release-audit", required=True, type=Path)
    parser.add_argument("--bundle-verification", required=True, type=Path)
    parser.add_argument(
        "--candidate-report",
        action="append",
        default=[],
        type=Path,
        help="Optional M6/M7 candidate report; may be repeated.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = audit_readiness(
        checkpoint_audit=args.checkpoint_audit,
        run_audit=args.run_audit,
        release_audit=args.release_audit,
        bundle_verification=args.bundle_verification,
        candidate_reports=args.candidate_report,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[release-readiness] {report['status'].upper()}")
    print(f"[release-readiness] report written: {output}")
    print(f"[release-readiness] next gate: {report['next_gate']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
