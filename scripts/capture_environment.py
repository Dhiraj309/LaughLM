#!/usr/bin/env python3
"""Capture a reproducible LaughLM runtime environment manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from LaughLM.provenance.environment import (
    build_runtime_manifest,
    write_runtime_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write a versioned LaughLM runtime environment manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON output path. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config whose exact bytes are SHA-256 hashed.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Optional repository root used for Git provenance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = build_runtime_manifest(
            repo_root=args.repo_root,
            config_path=args.config,
        )
        output = write_runtime_manifest(manifest, args.output)
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        _parser().error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
