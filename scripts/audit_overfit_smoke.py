#!/usr/bin/env python3
"""Audit a fixed-batch overfit smoke run from saved artifacts only."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_losses(path: Path) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            loss = row.get("loss") if isinstance(row, dict) else None
            if isinstance(loss, (int, float)) and not isinstance(loss, bool):
                values.append(float(loss))
    return values


def audit_overfit_smoke(
    *,
    manifest_path: Path,
    metrics_path: Path,
    min_loss_drop: float,
    max_final_loss: float | None,
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    gate = manifest.get("training_gates", {}).get("overfit_smoke", {})
    losses = _load_losses(metrics_path)
    finite = bool(losses) and all(math.isfinite(value) for value in losses)
    first_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None
    loss_drop = (first_loss - final_loss) if first_loss is not None and final_loss is not None else None
    checks = [
        {
            "name": "fixed-batch gate declaration",
            "passed": isinstance(gate, dict) and gate.get("enabled") is True and gate.get("mode") == "fixed_batch_v1" and bool(gate.get("batch_checksum")),
            "expected": "enabled fixed_batch_v1 gate with checksum",
            "actual": gate,
        },
        {
            "name": "finite loss series",
            "passed": finite,
            "expected": "at least two finite loss values",
            "actual": {"rows": len(losses), "finite": finite},
        },
        {
            "name": "fixed-batch loss improvement",
            "passed": loss_drop is not None and len(losses) >= 2 and loss_drop >= float(min_loss_drop),
            "expected": f"loss drop >= {min_loss_drop}",
            "actual": {"first": first_loss, "final": final_loss, "drop": loss_drop},
        },
    ]
    if max_final_loss is not None:
        checks.append({
            "name": "final loss ceiling",
            "passed": final_loss is not None and final_loss <= max_final_loss,
            "expected": f"final loss <= {max_final_loss}",
            "actual": final_loss,
        })
    passed = all(check["passed"] for check in checks)
    return {"audit": "LaughLM fixed-batch overfit smoke", "status": "pass" if passed else "fail", "manifest_path": str(manifest_path), "metrics_path": str(metrics_path), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit fixed-batch overfit evidence without runtime imports.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--min-loss-drop", type=float, default=1.0)
    parser.add_argument("--max-final-loss", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_overfit_smoke(
        manifest_path=args.manifest.expanduser().resolve(),
        metrics_path=args.metrics.expanduser().resolve(),
        min_loss_drop=args.min_loss_drop,
        max_final_loss=args.max_final_loss,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[overfit-audit] report written: {output}")
    print(f"[overfit-audit] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
