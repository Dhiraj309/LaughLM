#!/usr/bin/env python3
"""Audit LaughLM dataset and split provenance without runtime imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from LaughLM.data.manifest_contract import validate_artifact_contract

TOKEN_STORAGE = {"uint16": 2, "uint64": 8}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _record(checks: list[dict[str, Any]], name: str, passed: bool, expected: Any, actual: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "expected": expected, "actual": actual})


def audit_dataset_contract(manifest_path: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    checks: list[dict[str, Any]] = []
    contract_errors = validate_artifact_contract(manifest.get("artifact_contract"))
    _record(checks, "artifact contract", not contract_errors, "valid training artifact contract", contract_errors or "valid")

    data = manifest.get("data")
    config = manifest.get("resolved_config")
    model = config.get("model", {}) if isinstance(config, dict) else {}
    tokenizer = config.get("tokenizer", {}) if isinstance(config, dict) else {}
    train_files = data.get("train_files", []) if isinstance(data, dict) else []
    validation_files = data.get("validation_files", []) if isinstance(data, dict) else []
    _record(checks, "explicit train and validation paths", isinstance(train_files, list) and bool(train_files) and isinstance(validation_files, list), "train and validation path lists", {"train": train_files, "validation": validation_files})
    overlap = sorted(set(train_files).intersection(validation_files)) if isinstance(train_files, list) and isinstance(validation_files, list) else []
    _record(checks, "train/validation shard separation", not overlap, "no shared shard paths", overlap)
    vocab_size = model.get("vocab_size")
    tokenizer_vocab = tokenizer.get("vocab_size")
    _record(checks, "tokenizer/model vocabulary agreement", vocab_size == tokenizer_vocab, "model.vocab_size == tokenizer.vocab_size", {"model": vocab_size, "tokenizer": tokenizer_vocab})
    token_dtype = data.get("token_dtype") if isinstance(data, dict) else None
    itemsize = data.get("token_itemsize_bytes") if isinstance(data, dict) else None
    _record(checks, "token storage agreement", token_dtype in TOKEN_STORAGE and itemsize == TOKEN_STORAGE.get(token_dtype), "uint16/uint64 with matching itemsize", {"dtype": token_dtype, "itemsize": itemsize})

    shard_errors: list[str] = []
    for split in ("train", "validation"):
        details = data.get(f"{split}_shard_details", []) if isinstance(data, dict) else []
        if not isinstance(details, list) or not details:
            shard_errors.append(f"{split}: missing details")
            continue
        for index, detail in enumerate(details):
            if not isinstance(detail, dict):
                shard_errors.append(f"{split}[{index}]: not an object")
                continue
            if detail.get("dtype") != token_dtype or detail.get("itemsize_bytes") != itemsize:
                shard_errors.append(f"{split}[{index}]: storage mismatch")
            expected_bytes = (
                detail.get("token_count", 0) * itemsize
                if isinstance(itemsize, int) and isinstance(detail.get("token_count"), int)
                else None
            )
            if detail.get("size_aligned") is not True or detail.get("byte_size") != expected_bytes:
                shard_errors.append(f"{split}[{index}]: byte/token count mismatch")
            if not detail.get("path") or not detail.get("filename"):
                shard_errors.append(f"{split}[{index}]: missing path identity")
    _record(checks, "shard contract", not shard_errors, "aligned non-empty shard details", shard_errors or "valid")

    exposure = data.get("exposure") if isinstance(data, dict) else None
    exposure_enabled = isinstance(exposure, dict) and bool(exposure.get("enabled"))
    exposure_errors: list[str] = []
    if exposure_enabled:
        for split in ("train", "validation"):
            stats = exposure.get(split)
            if stats is None and split == "validation" and not validation_files:
                continue
            if not isinstance(stats, dict):
                exposure_errors.append(f"{split}: missing exposure stats")
                continue
            for field in ("total_tokens", "unique_token_count", "max_token_exposure", "frequency_checksum"):
                if field not in stats:
                    exposure_errors.append(f"{split}: missing {field}")
    _record(checks, "exposure statistics", (not exposure_enabled) or not exposure_errors, "disabled or complete per-split exposure statistics", {"enabled": exposure_enabled, "errors": exposure_errors})
    passed = all(item["passed"] for item in checks)
    return {"audit": "LaughLM dataset contract", "status": "pass" if passed else "fail", "manifest_path": str(manifest_path), "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit LaughLM dataset provenance without runtime imports.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_dataset_contract(args.manifest.expanduser().resolve())
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[dataset-contract] report written: {output}")
    print(f"[dataset-contract] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
