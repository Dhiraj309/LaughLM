#!/usr/bin/env python3
"""Audit LaughLM release artifacts without importing JAX or model code.

The audit checks the configuration, exported Hugging Face metadata, source
checkpoint manifest, and saved benchmark report. It intentionally performs no
checkpoint restore, model construction, tensor loading, or accelerator work.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SPECIAL_TOKEN_CONTRACT = {
    "bos_token_id": 1,
    "eos_token_id": 32000,
    "pad_token_id": 32000,
}


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


def _record(
    checks: list[dict[str, Any]],
    name: str,
    *,
    passed: bool,
    expected: Any = None,
    actual: Any = None,
    detail: str = "",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "expected": expected,
            "actual": actual,
            "detail": detail,
        }
    )


def audit_release(
    *,
    config_path: Path,
    export_dir: Path,
    run_manifest_path: Path,
    benchmark_report: Path,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config = _load_yaml(config_path)
    model = config.get("model", {})
    architecture = config.get("architecture", {})
    tokenizer = config.get("tokenizer", {})

    vocab_size = _nested(config, "model", "vocab_size")
    tokenizer_vocab_size = _nested(config, "tokenizer", "vocab_size")
    _record(
        checks,
        "config/tokenizer vocabulary size",
        passed=vocab_size == tokenizer_vocab_size,
        expected=vocab_size,
        actual=tokenizer_vocab_size,
    )

    num_heads = _nested(config, "model", "num_heads")
    num_kv_heads = _nested(config, "model", "num_kv_heads")
    attention_variant = _nested(config, "architecture", "attention_variant")
    if num_kv_heads is None:
        num_kv_heads = num_heads

    geometry_valid = (
        isinstance(num_heads, int)
        and isinstance(num_kv_heads, int)
        and num_heads > 0
        and num_kv_heads > 0
        and num_heads % num_kv_heads == 0
        and (
            (attention_variant == "mha" and num_kv_heads == num_heads)
            or (
                attention_variant == "gqa"
                and num_kv_heads < num_heads
            )
            or (attention_variant == "mqa" and num_kv_heads == 1)
        )
    )
    _record(
        checks,
        "attention geometry",
        passed=geometry_valid,
        expected="valid MHA/GQA/MQA geometry",
        actual={
            "variant": attention_variant,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
        },
    )

    for name, expected in SPECIAL_TOKEN_CONTRACT.items():
        configured = _nested(config, "model", name)
        if configured is None:
            configured = expected
        _record(
            checks,
            f"configured {name}",
            passed=configured == expected,
            expected=expected,
            actual=configured,
        )

    required_files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "source_checkpoint_metadata.json",
    ]
    for filename in required_files:
        path = export_dir / filename
        _record(
            checks,
            f"export file: {filename}",
            passed=path.is_file(),
            expected="file exists",
            actual=str(path) if path.is_file() else "missing",
        )

    tokenizer_files_present = [
        filename
        for filename in ("tokenizer.json", "tokenizer.model")
        if (export_dir / filename).is_file()
    ]
    _record(
        checks,
        "export tokenizer payload",
        passed=bool(tokenizer_files_present),
        expected="tokenizer.json or tokenizer.model",
        actual=tokenizer_files_present,
    )

    exported_config_path = export_dir / "config.json"
    generation_config_path = export_dir / "generation_config.json"
    source_metadata_path = export_dir / "source_checkpoint_metadata.json"

    exported_config = (
        _load_json(exported_config_path)
        if exported_config_path.is_file()
        else {}
    )
    generation_config = (
        _load_json(generation_config_path)
        if generation_config_path.is_file()
        else {}
    )
    source_metadata = (
        _load_json(source_metadata_path)
        if source_metadata_path.is_file()
        else {}
    )

    exported_checks = {
        "vocab_size": vocab_size,
        "num_key_value_heads": num_kv_heads,
        "tie_word_embeddings": bool(
            _nested(config, "architecture", "weight_tying")
        ),
        **SPECIAL_TOKEN_CONTRACT,
    }
    for name, expected in exported_checks.items():
        _record(
            checks,
            f"HF config {name}",
            passed=exported_config.get(name) == expected,
            expected=expected,
            actual=exported_config.get(name),
        )

        _record(
            checks,
            f"generation config {name}",
            passed=generation_config.get(name) == expected,
            expected=expected,
            actual=generation_config.get(name),
        )

    source_model = source_metadata.get("model", {})
    source_architecture = source_metadata.get("architecture", {})
    for name, expected in (
        ("vocab_size", vocab_size),
        ("num_heads", num_heads),
        ("num_kv_heads", num_kv_heads),
        ("attention_variant", attention_variant),
    ):
        _record(
            checks,
            f"source metadata model.{name}",
            passed=isinstance(source_model, dict)
            and source_model.get(name) == expected,
            expected=expected,
            actual=(
                source_model.get(name)
                if isinstance(source_model, dict)
                else None
            ),
        )

    expected_tying = bool(_nested(config, "architecture", "weight_tying"))
    _record(
        checks,
        "source metadata architecture.weight_tying",
        passed=isinstance(source_architecture, dict)
        and source_architecture.get("weight_tying") == expected_tying,
        expected=expected_tying,
        actual=(
            source_architecture.get("weight_tying")
            if isinstance(source_architecture, dict)
            else None
        ),
    )

    manifest = _load_json(run_manifest_path) if run_manifest_path.is_file() else {}
    _record(
        checks,
        "run manifest exists",
        passed=bool(manifest),
        expected="JSON run manifest",
        actual=str(run_manifest_path) if manifest else "missing",
    )
    _record(
        checks,
        "run manifest git revision",
        passed=bool(manifest.get("git_revision")),
        expected="non-empty git revision",
        actual=manifest.get("git_revision"),
    )
    package_versions = manifest.get("package_versions")
    _record(
        checks,
        "run manifest dependency versions",
        passed=isinstance(package_versions, dict)
        and all(
            package_versions.get(name)
            for name in ("jax", "jaxlib", "flax", "optax", "orbax-checkpoint")
        ),
        expected="core package versions recorded",
        actual=package_versions,
    )

    resolved_config = manifest.get("resolved_config", {})
    manifest_hf_repo = _nested(resolved_config, "data", "hf_repo_id")
    manifest_hf_revision = _nested(resolved_config, "data", "hf_revision")
    _record(
        checks,
        "HF dataset provenance",
        passed=bool(manifest_hf_repo),
        expected="repo ID and revision/default branch recorded",
        actual={
            "hf_repo_id": manifest_hf_repo,
            "hf_revision": manifest_hf_revision or "default branch",
        },
    )

    _record(
        checks,
        "benchmark report exists",
        passed=benchmark_report.is_file(),
        expected="saved benchmark report",
        actual=str(benchmark_report) if benchmark_report.is_file() else "missing",
    )

    passed = all(check["passed"] for check in checks)
    return {
        "audit": "LaughLM release contract",
        "status": "pass" if passed else "fail",
        "config_path": str(config_path),
        "export_dir": str(export_dir),
        "run_manifest": str(run_manifest_path),
        "benchmark_report": str(benchmark_report),
        "release_identity": {
            "vocab_size": vocab_size,
            "attention_variant": attention_variant,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "weight_tying": bool(_nested(config, "architecture", "weight_tying")),
            "hf_repo_id": manifest_hf_repo,
            "hf_revision": manifest_hf_revision or "default branch",
            "git_revision": manifest.get("git_revision"),
        },
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LaughLM release artifacts without runtime execution."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--benchmark-report", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("release_audit.json"))
    args = parser.parse_args()

    run_manifest = args.run_manifest
    if run_manifest is None and args.checkpoint_dir is not None:
        run_manifest = args.checkpoint_dir / "run_manifest.json"
    if run_manifest is None:
        parser.error("provide --run-manifest or --checkpoint-dir")

    report = audit_release(
        config_path=args.config,
        export_dir=args.export_dir,
        run_manifest_path=run_manifest,
        benchmark_report=args.benchmark_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[release-audit] {report['status'].upper()}")
    print(f"[release-audit] report written: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
