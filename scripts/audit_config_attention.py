#!/usr/bin/env python3
"""Audit attention labels and head geometry in maintained YAML configs.

This is a dependency-light static audit. It intentionally imports neither JAX
nor LaughLM model code, so it can run before reserving TPU time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def audit_config(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    model = _mapping(document.get("model"))
    architecture = _mapping(document.get("architecture"))
    variant = architecture.get("attention_variant")
    num_heads = model.get("num_heads")
    num_kv_heads = model.get("num_kv_heads")

    # Overlay files intentionally contain only changed fields. They are
    # validated after merging with their base config by the normal loader.
    if variant is None or num_heads is None:
        return []

    if num_kv_heads is None:
        num_kv_heads = num_heads

    errors = []
    if variant == "mha" and num_kv_heads != num_heads:
        errors.append(
            f"{path}: attention_variant='mha' requires "
            f"num_kv_heads={num_heads}, got {num_kv_heads}"
        )
    elif variant == "gqa":
        if num_kv_heads >= num_heads:
            errors.append(
                f"{path}: attention_variant='gqa' requires fewer KV heads "
                f"than query heads, got {num_kv_heads}/{num_heads}"
            )
        elif num_heads % num_kv_heads != 0:
            errors.append(
                f"{path}: GQA query heads must be divisible by KV heads, "
                f"got {num_heads}/{num_kv_heads}"
            )
    elif variant == "mqa" and num_kv_heads != 1:
        errors.append(
            f"{path}: attention_variant='mqa' requires num_kv_heads=1, "
            f"got {num_kv_heads}"
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit attention variant/head geometry in YAML configs."
    )
    parser.add_argument(
        "config_dir",
        nargs="?",
        default="configs",
        help="Directory containing YAML config files.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir).expanduser()
    paths = sorted(
        path
        for pattern in ("*.yaml", "*.yml")
        for path in config_dir.glob(pattern)
    )

    if not paths:
        print(f"[config-audit] no YAML configs found in {config_dir}")
        return 1

    errors = [error for path in paths for error in audit_config(path)]
    if errors:
        print("[config-audit] invalid attention geometry:")
        print("\n".join(f"  - {error}" for error in errors))
        return 1

    print(
        f"[config-audit] OK: checked {len(paths)} YAML configs; "
        "no invalid attention geometries found"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
