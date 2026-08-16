#!/usr/bin/env python3
"""Build a reproducible LaughLM release bundle without runtime imports.

This tool deliberately uses only standard-library modules and PyYAML. It does
not restore checkpoints, import JAX, construct a model, or load tensors. The
caller supplies the already-produced export, checkpoint evidence, benchmark
report, and optional TPU logs/profiles. Every copied file receives a SHA-256
record in ``release_manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


BUNDLE_FORMAT = "laughlm-release-bundle-v1"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _ensure_input_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} must be an existing file: {resolved}")
    return resolved


def _ensure_input_dir(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} must be an existing directory: {resolved}")
    return resolved


def _add_file(
    plan: list[tuple[Path, Path]],
    source: Path,
    destination: Path,
) -> None:
    if any(target == destination for _, target in plan):
        raise ValueError(f"Multiple inputs map to bundle path: {destination}")
    plan.append((source, destination))


def _add_directory(
    plan: list[tuple[Path, Path]],
    source_dir: Path,
    destination_dir: Path,
) -> int:
    count = 0
    for source in sorted(source_dir.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"Release inputs must not contain symlinks: {source}")
        if source.is_dir():
            continue
        if not source.is_file():
            raise ValueError(f"Unsupported non-file release input: {source}")
        relative = source.relative_to(source_dir)
        _add_file(plan, source, destination_dir / relative)
        count += 1
    return count


def _copy_plan(
    plan: list[tuple[Path, Path]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for source, relative_destination in sorted(plan, key=lambda item: str(item[1])):
        destination = output_dir / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative_destination.as_posix(),
                "source": str(source),
                "size_bytes": destination.stat().st_size,
                "sha256": _sha256(destination),
            }
        )
    return files


def _resolve_optional_file(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    return _ensure_input_file(path, label)


def build_bundle(
    *,
    config_path: Path,
    checkpoint_dir: Path,
    export_dir: Path,
    benchmark_report: Path,
    output_dir: Path,
    audit_report: Path | None,
    logs: list[Path],
    profiles: list[Path],
    force: bool,
) -> dict[str, Any]:
    config_path = _ensure_input_file(config_path, "--config")
    checkpoint_dir = _ensure_input_dir(checkpoint_dir, "--checkpoint-dir")
    export_dir = _ensure_input_dir(export_dir, "--export-dir")
    benchmark_report = _ensure_input_file(benchmark_report, "--benchmark-report")
    audit_report = _resolve_optional_file(audit_report, "--audit-report")
    logs = [_ensure_input_file(path, f"--log {path}") for path in logs]
    profiles = [_ensure_input_file(path, f"--profile {path}") for path in profiles]

    run_manifest = _ensure_input_file(
        checkpoint_dir / "run_manifest.json",
        "checkpoint run manifest",
    )
    metrics = _ensure_input_file(
        checkpoint_dir / "metrics.jsonl",
        "checkpoint metrics",
    )
    metadata_dir = checkpoint_dir / "checkpoint_metadata"
    metadata_files = []
    if metadata_dir.is_symlink():
        raise ValueError(f"Checkpoint metadata directory must not be a symlink: {metadata_dir}")
    if metadata_dir.is_dir():
        metadata_files = sorted(metadata_dir.glob("step_*.json"))
        metadata_files = [
            _ensure_input_file(path, "checkpoint metadata")
            for path in metadata_files
        ]
    if not metadata_files:
        raise FileNotFoundError(
            "checkpoint directory must contain checkpoint_metadata/step_*.json"
        )

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and not force:
        raise FileExistsError(
            f"Refusing to use existing output directory without --force: {output_dir}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir is not a directory: {output_dir}")

    for input_root in (checkpoint_dir, export_dir):
        if _is_within(output_dir, input_root) or _is_within(input_root, output_dir):
            raise ValueError(
                "--output-dir must be separate from --checkpoint-dir and "
                f"--export-dir: {output_dir}"
            )
    input_files = [
        config_path,
        benchmark_report,
        run_manifest,
        metrics,
        *metadata_files,
        *logs,
        *profiles,
    ]
    if audit_report is not None:
        input_files.append(audit_report)
    if any(_is_within(path, output_dir) for path in input_files):
        raise ValueError("An input file cannot be inside --output-dir")

    config = _load_yaml(config_path)
    manifest = _load_json(run_manifest)
    audit = _load_json(audit_report) if audit_report is not None else None
    if audit is not None and audit.get("status") != "pass":
        raise ValueError(
            f"--audit-report is not a passing audit: {audit.get('status')!r}"
        )

    plan: list[tuple[Path, Path]] = []
    _add_file(plan, config_path, Path("config") / config_path.name)
    _add_file(plan, run_manifest, Path("provenance") / "run_manifest.json")
    _add_file(plan, metrics, Path("provenance") / "metrics.jsonl")
    for metadata in metadata_files:
        _add_file(
            plan,
            metadata,
            Path("provenance") / "checkpoint_metadata" / metadata.name,
        )
    _add_directory(plan, export_dir, Path("export"))
    _add_file(
        plan,
        benchmark_report,
        Path("evidence") / "benchmarks" / benchmark_report.name,
    )
    if audit_report is not None:
        _add_file(plan, audit_report, Path("provenance") / "release_audit.json")
    for source in logs:
        _add_file(plan, source, Path("evidence") / "logs" / source.name)
    for source in profiles:
        _add_file(plan, source, Path("evidence") / "profiles" / source.name)

    output_dir.mkdir(parents=True, exist_ok=True)
    files = _copy_plan(plan, output_dir)

    resolved_data = _nested(manifest, "resolved_config", "data") or {}
    release_manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "config": str(config_path),
            "checkpoint_dir": str(checkpoint_dir),
            "export_dir": str(export_dir),
            "benchmark_report": str(benchmark_report),
            "run_manifest": str(run_manifest),
            "audit_report": str(audit_report) if audit_report else None,
            "logs": [str(path) for path in logs],
            "profiles": [str(path) for path in profiles],
        },
        "release_identity": {
            "vocab_size": _nested(config, "model", "vocab_size"),
            "attention_variant": _nested(
                config,
                "architecture",
                "attention_variant",
            ),
            "num_heads": _nested(config, "model", "num_heads"),
            "num_kv_heads": _nested(config, "model", "num_kv_heads"),
            "weight_tying": _nested(config, "architecture", "weight_tying"),
            "hf_repo_id": resolved_data.get("hf_repo_id"),
            "hf_revision": resolved_data.get("hf_revision") or "default branch",
            "git_revision": manifest.get("git_revision"),
        },
        "provenance": {
            "package_versions": manifest.get("package_versions", {}),
            "manifest_version": manifest.get("manifest_version"),
            "checkpoint_metadata_files": len(metadata_files),
            "audit_status": audit.get("status") if audit is not None else None,
        },
        "files": files,
    }
    manifest_path = output_dir / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a checksummed LaughLM release bundle without runtime imports."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument(
        "--log",
        action="append",
        default=[],
        type=Path,
        help="TPU log to archive; may be supplied more than once.",
    )
    parser.add_argument(
        "--profile",
        action="append",
        default=[],
        type=Path,
        help="Profiler artifact to archive; may be supplied more than once.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow an existing output directory; existing files may be overwritten.",
    )
    args = parser.parse_args()

    manifest = build_bundle(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        export_dir=args.export_dir,
        benchmark_report=args.benchmark_report,
        output_dir=args.output_dir,
        audit_report=args.audit_report,
        logs=args.log,
        profiles=args.profile,
        force=args.force,
    )
    resolved_output = args.output_dir.expanduser().resolve()
    print(
        f"[release-bundle] wrote {len(manifest['files'])} files to {resolved_output}"
    )
    print(f"[release-bundle] manifest: {resolved_output / 'release_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
