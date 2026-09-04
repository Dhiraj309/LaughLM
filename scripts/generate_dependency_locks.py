#!/usr/bin/env python3
"""Generate deterministic, hash-complete locks with pip-tools.

This wrapper owns the reproducibility policy. pip-compile remains the resolver;
LaughLM validates its output, removes transient headers, and writes a stable
project header containing the exact generator version and resolver mode.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOLVER = "backtracking"
PACKAGE_LINE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)==(?P<version>[^\s;]+)"
)
HASH_LINE = re.compile(r"--hash=sha256:[0-9a-f]{64}(?:\s+\\)?$")


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _pip_tools_version() -> str:
    try:
        return importlib.metadata.version("pip-tools")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "pip-tools is required; install requirements/inputs/lock-tools-py312.in"
        ) from exc


def validate_compiled_lock(compiled_text: str) -> list[str]:
    """Validate pip-compile's hash output and return normalized package names."""

    package_names: list[str] = []
    current_name: str | None = None
    current_hashes = 0

    def finish_current() -> None:
        nonlocal current_name, current_hashes
        if current_name is None:
            return
        if current_hashes == 0:
            raise ValueError(
                f"Lock entry {current_name!r} has no SHA-256 artifact hash"
            )
        current_name = None
        current_hashes = 0

    for line_number, raw_line in enumerate(compiled_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            if current_name is None:
                raise ValueError(f"Hash without a package entry on line {line_number}")
            if HASH_LINE.fullmatch(line) is None:
                raise ValueError(f"Invalid SHA-256 hash on line {line_number}")
            current_hashes += 1
            continue
        if line.startswith("--"):
            raise ValueError(
                f"Lock contains an unpinned pip option on line {line_number}: {line}"
            )

        match = PACKAGE_LINE.match(line)
        if match is None:
            raise ValueError(
                "Lock contains a non-exact or unsupported requirement on "
                f"line {line_number}: {line}"
            )

        finish_current()
        current_name = _normalized_name(match.group("name"))
        if current_name in package_names:
            raise ValueError(f"Duplicate package entry: {current_name}")
        package_names.append(current_name)
        inline_hashes = HASH_LINE.findall(line)
        current_hashes = len(inline_hashes)

    finish_current()
    if not package_names:
        raise ValueError("Generated lock contains no package entries")
    return package_names


def deterministic_lock_text(
    compiled_text: str,
    *,
    input_path: Path,
    repo_root: Path,
    generator_version: str,
    resolver: str = DEFAULT_RESOLVER,
) -> str:
    """Return canonical lock text with a stable provenance header."""

    validate_compiled_lock(compiled_text)
    lines = [line.rstrip() for line in compiled_text.replace("\r\n", "\n").split("\n")]
    first_requirement = next(
        (
            index
            for index, line in enumerate(lines)
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if first_requirement is None:
        raise ValueError("Generated lock contains no body")

    body = "\n".join(lines[first_requirement:]).strip() + "\n"
    header = "\n".join(
        [
            "# LaughLM generated dependency lock; do not edit by hand.",
            f"# input: {_relative_path(input_path, repo_root)}",
            f"# generator: pip-tools=={generator_version}",
            f"# resolver: {resolver}",
            "# hashes: sha256, required for every package entry",
            "",
        ]
    )
    return header + body


def _write_atomic(path: Path, text: str, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing lock: {path}; pass --force to replace it"
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def generate_lock(
    *,
    input_path: Path,
    output_path: Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    pip_compile: str = "pip-compile",
    resolver: str = DEFAULT_RESOLVER,
    force: bool = False,
) -> Path:
    """Resolve one input, validate hashes, and atomically write its lock."""

    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    repo_root = repo_root.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Dependency input not found: {input_path}")
    if not (repo_root / "pyproject.toml").is_file():
        raise ValueError(f"Repository root does not contain pyproject.toml: {repo_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing lock: {output_path}; pass --force to replace it"
        )

    executable = shutil.which(pip_compile) or pip_compile
    generator_version = _pip_tools_version()
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".compiled.tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

        command = [
            executable,
            "--generate-hashes",
            f"--resolver={resolver}",
            "--allow-unsafe",
            f"--output-file={temporary_path}",
            _relative_path(input_path, repo_root),
        ]
        result = subprocess.run(
            command,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(
                f"pip-compile failed with exit code {result.returncode}: {detail}"
            )

        compiled_text = temporary_path.read_text(encoding="utf-8")
        lock_text = deterministic_lock_text(
            compiled_text,
            input_path=input_path,
            repo_root=repo_root,
            generator_version=generator_version,
            resolver=resolver,
        )
        _write_atomic(output_path, lock_text, force=force)
        return output_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate one deterministic, hash-complete LaughLM lock."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
    )
    parser.add_argument(
        "--pip-compile",
        default="pip-compile",
        help="pip-compile executable name or absolute path",
    )
    parser.add_argument(
        "--resolver",
        choices=("backtracking",),
        default=DEFAULT_RESOLVER,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated lock atomically",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = generate_lock(
            input_path=args.input,
            output_path=args.output,
            repo_root=args.repo_root,
            pip_compile=args.pip_compile,
            resolver=args.resolver,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        _parser().error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
