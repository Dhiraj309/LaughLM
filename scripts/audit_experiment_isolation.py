#!/usr/bin/env python3
"""Audit LaughLM YAML experiment overlays for artifact isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


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


def audit_isolation(
    *,
    base_config_path: Path,
    overlay_paths: list[Path],
) -> dict[str, Any]:
    base_config = _load_yaml(base_config_path)
    base_checkpoint = _nested(base_config, "runtime", "checkpoint_dir")
    base_cache = _nested(base_config, "optimizations", "compilation_cache_dir")
    checks: list[dict[str, Any]] = []

    _record(
        checks,
        "base checkpoint directory",
        passed=bool(base_checkpoint),
        expected="non-empty runtime.checkpoint_dir",
        actual=base_checkpoint,
    )
    _record(
        checks,
        "base compilation cache directory",
        passed=bool(base_cache),
        expected="non-empty optimizations.compilation_cache_dir",
        actual=base_cache,
    )

    seen_checkpoints: dict[str, str] = {}
    seen_caches: dict[str, str] = {}
    seen_profiles: dict[str, str] = {}
    overlay_results: list[dict[str, Any]] = []
    for path in overlay_paths:
        config = _load_yaml(path)
        checkpoint = _nested(config, "runtime", "checkpoint_dir")
        cache = _nested(config, "optimizations", "compilation_cache_dir")
        profile_output = _nested(config, "profiling", "output_dir")
        local_checks: list[dict[str, Any]] = []

        _record(
            local_checks,
            "checkpoint directory override",
            passed=bool(checkpoint) and checkpoint != base_checkpoint,
            expected="non-empty path different from base",
            actual=checkpoint,
        )
        _record(
            local_checks,
            "compilation cache override",
            passed=bool(cache) and cache != base_cache,
            expected="non-empty path different from base",
            actual=cache,
        )
        if checkpoint:
            previous = seen_checkpoints.get(str(checkpoint))
            _record(
                local_checks,
                "unique checkpoint directory",
                passed=previous is None,
                expected="not used by another overlay",
                actual=previous or checkpoint,
            )
            seen_checkpoints[str(checkpoint)] = path.name
        if cache:
            previous = seen_caches.get(str(cache))
            _record(
                local_checks,
                "unique compilation cache",
                passed=previous is None,
                expected="not used by another overlay",
                actual=previous or cache,
            )
            seen_caches[str(cache)] = path.name
        if profile_output:
            previous = seen_profiles.get(str(profile_output))
            _record(
                local_checks,
                "unique profiling output",
                passed=previous is None,
                expected="not used by another overlay",
                actual=previous or profile_output,
            )
            seen_profiles[str(profile_output)] = path.name

        overlay_results.append(
            {
                "overlay": str(path),
                "checkpoint_dir": checkpoint,
                "compilation_cache_dir": cache,
                "profiling_output_dir": profile_output,
                "status": "pass" if all(item["passed"] for item in local_checks) else "fail",
                "checks": local_checks,
            }
        )

    checks.append(
        {
            "name": "overlay isolation",
            "passed": all(item["status"] == "pass" for item in overlay_results),
            "expected": "every overlay has isolated artifacts",
            "actual": overlay_results,
        }
    )
    passed = all(check["passed"] for check in checks)
    return {
        "audit": "LaughLM experiment isolation",
        "status": "pass" if passed else "fail",
        "base_config": str(base_config_path),
        "overlay_count": len(overlay_paths),
        "overlays": overlay_results,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaughLM experiment overlay artifact isolation."
    )
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--overlay-dir", required=True, type=Path)
    parser.add_argument("--pattern", default="*override.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base_config = args.base_config.expanduser().resolve()
    overlay_dir = args.overlay_dir.expanduser().resolve()
    overlays = sorted(overlay_dir.glob(args.pattern))
    if not overlays:
        parser.error(f"no overlays matched {overlay_dir / args.pattern}")

    report = audit_isolation(
        base_config_path=base_config,
        overlay_paths=overlays,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[experiment-isolation] report written: {output}")
    print(f"[experiment-isolation] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
