#!/usr/bin/env python3
"""Audit checkpoint/config compatibility without importing runtime code.

This is a preflight aid for deliberate resume-rejection checks. It compares
LaughLM v3 JSON metadata with the current YAML contract and never restores a
checkpoint, imports JAX/Orbax, or inspects model tensors. Runtime restore still
performs the authoritative compatibility validation, including full layout
checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


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


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None and default is not None else value


def _canonical_backend(config: dict[str, Any]) -> str:
    backend = _nested(config, "runtime", "backend", default="pmap")
    return "fsdp" if backend == "gspmd" else backend


def _config_contract(config: dict[str, Any]) -> dict[str, Any]:
    backend = _canonical_backend(config)
    return {
        "backend": backend,
        "model": {
            key: _nested(config, "model", key)
            for key in (
                "vocab_size",
                "d_model",
                "num_layers",
                "num_heads",
                "num_kv_heads",
                "max_seq_len",
            )
        },
        "architecture": {
            "attention_impl": _nested(
                config, "architecture", "attention_impl"
            ),
            "attention_fallback": _nested(
                config,
                "architecture",
                "attention_fallback",
                default="warn",
            ),
            "attention_variant": _nested(
                config, "architecture", "attention_variant"
            ),
            "fused_qkv": _nested(
                config, "architecture", "fused_qkv", default=False
            ),
            "weight_tying": _nested(
                config, "architecture", "weight_tying", default=True
            ),
        },
        "runtime": {
            "backend": _nested(config, "runtime", "backend", default="pmap"),
            "canonical_backend": backend,
            "seq_len": _nested(config, "runtime", "seq_len"),
            "micro_batch_per_device": _nested(
                config, "runtime", "micro_batch_per_device"
            ),
            "gradient_accumulation": _nested(
                config, "runtime", "gradient_accumulation"
            ),
            "total_tokens": _nested(config, "runtime", "total_tokens"),
        },
        "optimizer": {
            key: _nested(config, "optimizer", key)
            for key in (
                "type",
                "learning_rate",
                "beta1",
                "beta2",
                "eps",
                "weight_decay",
                "gradient_clip",
                "mu_dtype",
            )
        },
        "scheduler": {
            key: _nested(config, "scheduler", key)
            for key in (
                "type",
                "horizon_tokens",
                "warmup_fraction",
                "stable_fraction",
                "min_lr_ratio",
            )
        },
        "parallelism": {
            key: _nested(config, "parallelism", key)
            for key in (
                "data_parallel",
                "model_parallel",
                "compute_dtype",
                "param_dtype",
            )
        },
        "dtype_policy": {
            "spmd": {
                key: _nested(config, "spmd", "dtype", key)
                for key in ("param_dtype", "compute_dtype", "output_dtype")
            },
            "parallelism": {
                key: _nested(config, "parallelism", key)
                for key in ("param_dtype", "compute_dtype")
            },
        },
        "execution_contract": {
            "backend": backend,
            "effective_sharding_strategy": (
                backend
                if backend in {"pmap", "fsdp"}
                else _nested(
                    config,
                    "optimizations",
                    "sharding_strategy",
                )
            ),
            "attention_impl": _nested(config, "architecture", "attention_impl"),
            "attention_variant": _nested(
                config, "architecture", "attention_variant"
            ),
            "num_heads": _nested(config, "model", "num_heads"),
            "num_kv_heads": _nested(config, "model", "num_kv_heads"),
            "param_dtype": _nested(config, "spmd", "dtype", "param_dtype"),
            "compute_dtype": _nested(config, "spmd", "dtype", "compute_dtype"),
            "output_dtype": _nested(config, "spmd", "dtype", "output_dtype"),
            "remat_policy": _nested(config, "spmd", "remat", "policy"),
            "remat_granularity": _nested(
                config, "spmd", "remat", "granularity"
            ),
            "scan_layers": _nested(
                config, "spmd", "remat", "scan_layers", default=False
            ),
            "prevent_cse": _nested(
                config, "spmd", "remat", "prevent_cse", default=False
            ),
            "chunked_logits": _nested(
                config, "loss", "chunked_logits", default=False
            ),
            "logits_chunk_size": _nested(
                config, "loss", "logits_chunk_size", default=4096
            ),
            "loss_backend": _nested(
                config, "loss", "backend", default="native"
            ),
            "tokamax_implementation": _nested(
                config,
                "loss",
                "tokamax_implementation",
                default="mosaic_tpu",
            ),
        },
    }


def _metadata_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend": metadata.get("backend"),
        "model": metadata.get("model", {}),
        "architecture": {
            key: _nested(metadata, "architecture", key)
            for key in (
                "attention_impl",
                "attention_fallback",
                "attention_variant",
                "fused_qkv",
                "weight_tying",
            )
        },
        "runtime": metadata.get("runtime", {}),
        "optimizer": metadata.get("optimizer", {}),
        "scheduler": metadata.get("scheduler", {}),
        "parallelism": metadata.get("parallelism", {}),
        "dtype_policy": metadata.get("dtype_policy"),
        "execution_contract": metadata.get("execution_contract"),
    }


def _compare(
    checks: list[dict[str, Any]],
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": expected == actual,
            "expected": expected,
            "actual": actual,
        }
    )


def _compare_mapping(
    checks: list[dict[str, Any]],
    name: str,
    expected: Any,
    actual: Any,
) -> None:
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        _compare(checks, name, expected, actual)
        return
    for key, expected_value in expected.items():
        _compare(checks, f"{name}.{key}", expected_value, actual.get(key))


def audit_compatibility(
    *,
    config_path: Path,
    metadata_path: Path,
    require_v3: bool,
    expected_num_devices: int | None,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    metadata = _load_json(metadata_path)
    checks: list[dict[str, Any]] = []

    metadata_format = metadata.get("format")
    _compare(
        checks,
        "metadata format",
        "laughlm_checkpoint_v3" if require_v3 else metadata_format,
        metadata_format,
    )

    expected = _config_contract(config)
    actual = _metadata_contract(metadata)
    for section in (
        "backend",
        "model",
        "architecture",
        "runtime",
        "optimizer",
        "scheduler",
        "parallelism",
        "dtype_policy",
        "execution_contract",
    ):
        _compare_mapping(checks, section, expected[section], actual[section])

    if expected_num_devices is not None:
        _compare(
            checks,
            "num_devices",
            expected_num_devices,
            metadata.get("num_devices"),
        )

    passed = all(check["passed"] for check in checks)
    return {
        "audit": "LaughLM checkpoint/config compatibility",
        "status": "pass" if passed else "fail",
        "config_path": str(config_path),
        "metadata_path": str(metadata_path),
        "metadata_step": metadata.get("step"),
        "require_v3": require_v3,
        "expected_num_devices": expected_num_devices,
        "checks": checks,
    }


def _latest_metadata(checkpoint_dir: Path) -> Path:
    paths = sorted((checkpoint_dir / "checkpoint_metadata").glob("step_*.json"))
    if not paths:
        raise FileNotFoundError(
            "No checkpoint_metadata/step_*.json files found in checkpoint directory"
        )
    return max(paths, key=lambda path: int(path.stem.removeprefix("step_")))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit checkpoint/config compatibility without runtime imports."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--expected-num-devices", type=int)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.metadata is not None and args.checkpoint_dir is not None:
        parser.error("provide only one of --metadata or --checkpoint-dir")
    if args.metadata is None and args.checkpoint_dir is None:
        parser.error("provide --metadata or --checkpoint-dir")
    if args.step is not None and args.checkpoint_dir is None:
        parser.error("--step requires --checkpoint-dir")

    if args.metadata is not None:
        metadata_path = args.metadata
    elif args.step is not None:
        metadata_path = (
            args.checkpoint_dir
            / "checkpoint_metadata"
            / f"step_{args.step:08d}.json"
        )
    else:
        metadata_path = _latest_metadata(args.checkpoint_dir)

    report = audit_compatibility(
        config_path=args.config.expanduser().resolve(),
        metadata_path=metadata_path.expanduser().resolve(),
        require_v3=not args.allow_legacy,
        expected_num_devices=args.expected_num_devices,
    )
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[checkpoint-compatibility] report written: {output}")
    print(f"[checkpoint-compatibility] {report['status'].upper()}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
