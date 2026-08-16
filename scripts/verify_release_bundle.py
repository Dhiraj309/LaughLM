#!/usr/bin/env python3
"""Verify a LaughLM release bundle without importing runtime code.

The verifier reads ``release_manifest.json`` and checks the recorded file
paths, sizes, SHA-256 digests, required release contents, and provenance. It
does not restore checkpoints, import JAX, construct a model, or load tensors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


BUNDLE_FORMAT = "laughlm-release-bundle-v1"
CORE_PACKAGES = ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint")
REQUIRED_FILES = (
    "provenance/run_manifest.json",
    "provenance/metrics.jsonl",
)
REQUIRED_EXPORT_FILES = (
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "source_checkpoint_metadata.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        return None
    normalized = path.as_posix()
    return normalized if normalized == value else None


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


def verify_bundle(
    *,
    bundle_dir: Path,
    require_audit: bool,
    require_parity: bool,
    output_path: Path | None,
) -> dict[str, Any]:
    bundle_dir = bundle_dir.expanduser().resolve()
    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise FileNotFoundError(f"Bundle directory is invalid: {bundle_dir}")

    manifest_path = bundle_dir / "release_manifest.json"
    manifest = _load_json(manifest_path)
    checks: list[dict[str, Any]] = []

    _record(
        checks,
        "bundle format",
        passed=manifest.get("bundle_format") == BUNDLE_FORMAT,
        expected=BUNDLE_FORMAT,
        actual=manifest.get("bundle_format"),
    )

    file_records = manifest.get("files")
    records_are_valid = isinstance(file_records, list)
    _record(
        checks,
        "file manifest shape",
        passed=records_are_valid,
        expected="list of file records",
        actual=type(file_records).__name__,
    )
    if not records_are_valid:
        file_records = []

    recorded_paths: set[str] = set()
    invalid_records: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for record in file_records:
        if not isinstance(record, dict):
            invalid_records.append("non-object record")
            continue
        relative_path = _safe_relative_path(record.get("path"))
        if relative_path is None or relative_path in recorded_paths:
            invalid_records.append(str(record.get("path")))
            continue
        recorded_paths.add(relative_path)
        path = bundle_dir.joinpath(*PurePosixPath(relative_path).parts)
        if path.is_symlink() or not path.is_file():
            mismatches.append(
                {
                    "path": relative_path,
                    "reason": "missing or symlinked file",
                }
            )
            continue
        expected_size = record.get("size_bytes")
        expected_hash = record.get("sha256")
        actual_size = path.stat().st_size
        actual_hash = _sha256(path)
        if expected_size != actual_size or expected_hash != actual_hash:
            mismatches.append(
                {
                    "path": relative_path,
                    "reason": "size or SHA-256 mismatch",
                    "expected_size_bytes": expected_size,
                    "actual_size_bytes": actual_size,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                }
            )

    _record(
        checks,
        "file record paths",
        passed=not invalid_records,
        expected="unique safe relative paths",
        actual=invalid_records or "valid",
    )
    _record(
        checks,
        "file checksums",
        passed=not mismatches,
        expected="recorded size and SHA-256 for every file",
        actual=mismatches or "all files match",
    )

    actual_bundle_paths = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_symlink() or path.is_file()
    }
    manifest_name = manifest_path.relative_to(bundle_dir).as_posix()
    actual_payload_paths = actual_bundle_paths - {manifest_name}
    extra_paths = sorted(actual_payload_paths - recorded_paths)
    missing_paths = sorted(recorded_paths - actual_payload_paths)
    _record(
        checks,
        "bundle file set",
        passed=not extra_paths and not missing_paths,
        expected="payload files exactly match release_manifest.json",
        actual={"extra": extra_paths, "missing": missing_paths},
    )

    for relative_path in REQUIRED_FILES:
        _record(
            checks,
            f"required file: {relative_path}",
            passed=relative_path in recorded_paths,
            expected="recorded payload file",
            actual=relative_path if relative_path in recorded_paths else "missing",
        )

    export_paths = {
        path for path in recorded_paths if path.startswith("export/")
    }
    for filename in REQUIRED_EXPORT_FILES:
        relative_path = f"export/{filename}"
        _record(
            checks,
            f"required export file: {filename}",
            passed=relative_path in export_paths,
            expected="recorded export file",
            actual=relative_path if relative_path in export_paths else "missing",
        )
    tokenizer_present = any(
        path in export_paths for path in ("export/tokenizer.json", "export/tokenizer.model")
    )
    _record(
        checks,
        "export tokenizer payload",
        passed=tokenizer_present,
        expected="export/tokenizer.json or export/tokenizer.model",
        actual="present" if tokenizer_present else "missing",
    )

    metadata_present = any(
        path.startswith("provenance/checkpoint_metadata/step_")
        and path.endswith(".json")
        for path in recorded_paths
    )
    _record(
        checks,
        "checkpoint metadata payload",
        passed=metadata_present,
        expected="at least one checkpoint metadata sidecar",
        actual="present" if metadata_present else "missing",
    )

    release_identity = manifest.get("release_identity")
    identity_valid = isinstance(release_identity, dict)
    if identity_valid:
        identity_fields = (
            "vocab_size",
            "attention_variant",
            "num_heads",
            "num_kv_heads",
            "git_revision",
            "hf_repo_id",
        )
        identity_valid = all(release_identity.get(field) not in (None, "") for field in identity_fields)
    _record(
        checks,
        "release identity provenance",
        passed=identity_valid,
        expected="model identity, git revision, and HF repository",
        actual=release_identity,
    )

    provenance = manifest.get("provenance")
    package_versions = provenance.get("package_versions") if isinstance(provenance, dict) else None
    versions_valid = isinstance(package_versions, dict) and all(
        package_versions.get(package) for package in CORE_PACKAGES
    )
    _record(
        checks,
        "dependency version provenance",
        passed=versions_valid,
        expected="core runtime package versions",
        actual=package_versions,
    )

    audit_path = bundle_dir / "provenance" / "release_audit.json"
    audit_status = None
    if audit_path.is_file():
        audit = _load_json(audit_path)
        audit_status = audit.get("status")
    audit_ok = audit_status == "pass"
    _record(
        checks,
        "release audit status",
        passed=audit_ok if require_audit or audit_path.exists() else True,
        expected="passing release audit when present/required",
        actual=audit_status or ("not supplied" if not require_audit else "missing"),
    )

    parity_path = bundle_dir / "provenance" / "hf_parity_report.json"
    parity_status = None
    if parity_path.is_file():
        parity = _load_json(parity_path)
        parity_status = parity.get("status")
    parity_ok = parity_status == "pass"
    _record(
        checks,
        "HF parity report status",
        passed=parity_ok if require_parity or parity_path.exists() else True,
        expected="passing HF parity report when present/required",
        actual=parity_status or ("not supplied" if not require_parity else "missing"),
    )

    passed = all(check["passed"] for check in checks)
    report = {
        "verification": "LaughLM release bundle",
        "status": "pass" if passed else "fail",
        "bundle_dir": str(bundle_dir),
        "manifest": str(manifest_path),
        "require_audit": require_audit,
        "require_parity": require_parity,
        "checks": checks,
    }
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["output"] = str(output_path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a LaughLM release bundle without runtime imports."
    )
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument(
        "--require-audit",
        action="store_true",
        help="Require provenance/release_audit.json with status=pass.",
    )
    parser.add_argument(
        "--require-parity",
        action="store_true",
        help="Require provenance/hf_parity_report.json with status=pass.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = verify_bundle(
        bundle_dir=args.bundle_dir,
        require_audit=args.require_audit,
        require_parity=args.require_parity,
        output_path=args.output,
    )
    print(f"[release-bundle-verify] {report['status'].upper()}")
    if args.output:
        print(f"[release-bundle-verify] report written: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
