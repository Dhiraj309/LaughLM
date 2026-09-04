"""Capture the immutable environment facts needed to reproduce a run.

The capture path is intentionally independent of the training loop.  JAX is
loaded lazily so the manifest tool can still describe a broken or incomplete
environment instead of failing before it records package and Git state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


RUNTIME_MANIFEST_SCHEMA = "laughlm_runtime_manifest_v1"
RUNTIME_MANIFEST_VERSION = 1
RUNTIME_PACKAGE_NAMES = (
    "jax",
    "jaxlib",
    "libtpu",
    "flax",
    "optax",
    "orbax-checkpoint",
    "tokamax",
    "xprof",
    "grain",
    "pydantic",
)

PackageVersionResolver = Callable[[str], str | None]
GitState = tuple[str | None, bool | None]


def _default_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _package_versions(
    resolver: PackageVersionResolver | None = None,
) -> dict[str, str | None]:
    resolve = resolver or _default_package_version
    return {
        package_name: resolve(package_name)
        for package_name in RUNTIME_PACKAGE_NAMES
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(repo_root: Path) -> GitState:
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None, None

    if revision_result.returncode != 0:
        return None, None

    revision = revision_result.stdout.strip() or None
    try:
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return revision, None

    if status_result.returncode != 0:
        return revision, None
    return revision, bool(status_result.stdout.strip())


def _call_if_callable(value: Any) -> Any:
    return value() if callable(value) else value


def _optional_int(value: Any) -> int | None:
    value = _call_if_callable(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _capture_jax(
    package_version: str | None,
    jax_module: Any | None = None,
) -> dict[str, Any]:
    if jax_module is None:
        try:
            import jax as jax_module
        except Exception:
            return {
                "available": False,
                "version": package_version,
                "backend": None,
                "x64_enabled": None,
                "devices": [],
            }

    try:
        devices = list(jax_module.devices())
    except Exception:
        return {
            "available": True,
            "version": package_version,
            "backend": None,
            "x64_enabled": None,
            "devices": [],
        }

    device_records: list[dict[str, Any]] = []
    for index, device in enumerate(devices):
        device_records.append(
            {
                "id": _optional_int(getattr(device, "id", index)),
                "platform": str(getattr(device, "platform", "unknown")),
                "kind": str(
                    getattr(device, "device_kind", getattr(device, "platform", "unknown"))
                ),
                "process_index": _optional_int(
                    getattr(device, "process_index", None)
                ),
            }
        )

    config = getattr(jax_module, "config", None)
    return {
        "available": True,
        "version": package_version,
        "backend": _call_if_callable(
            getattr(jax_module, "default_backend", None)
        ),
        "x64_enabled": (
            bool(getattr(config, "x64_enabled"))
            if config is not None and hasattr(config, "x64_enabled")
            else None
        ),
        "process_index": _optional_int(
            getattr(jax_module, "process_index", None)
        ),
        "process_count": _optional_int(
            getattr(jax_module, "process_count", None)
        ),
        "devices": device_records,
    }


def _captured_at(value: str | None) -> str:
    if value is not None:
        if not value.strip():
            raise ValueError("captured_at_utc must not be empty")
        return value
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def build_runtime_manifest(
    *,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    captured_at_utc: str | None = None,
    package_version_resolver: PackageVersionResolver | None = None,
    git_state: GitState | None = None,
    jax_module: Any | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe, versioned runtime environment manifest."""

    resolved_repo_root = (
        Path(repo_root).expanduser().resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[2]
    )

    resolved_config_path: str | None = None
    config_sha256: str | None = None
    if config_path is not None:
        config = Path(config_path).expanduser().resolve()
        if not config.is_file():
            raise FileNotFoundError(f"Config file not found: {config}")
        resolved_config_path = str(config)
        config_sha256 = _sha256_file(config)

    package_versions = _package_versions(package_version_resolver)
    revision, dirty = git_state or _git_state(resolved_repo_root)

    return {
        "manifest_schema": RUNTIME_MANIFEST_SCHEMA,
        "manifest_version": RUNTIME_MANIFEST_VERSION,
        "captured_at_utc": _captured_at(captured_at_utc),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "build": platform.python_build()[0],
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "package_versions": package_versions,
        "git_revision": revision,
        "git_dirty": dirty,
        "config_path": resolved_config_path,
        "config_sha256": config_sha256,
        "jax": _capture_jax(
            package_versions.get("jax"),
            jax_module=jax_module,
        ),
    }


def validate_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the required v1 contract without requiring jsonschema."""

    if manifest.get("manifest_schema") != RUNTIME_MANIFEST_SCHEMA:
        raise ValueError("Unsupported runtime manifest schema")
    if manifest.get("manifest_version") != RUNTIME_MANIFEST_VERSION:
        raise ValueError("Unsupported runtime manifest version")

    for key in ("captured_at_utc", "python", "platform", "package_versions", "jax"):
        if key not in manifest:
            raise ValueError(f"Runtime manifest is missing {key!r}")

    package_versions = manifest["package_versions"]
    if not isinstance(package_versions, Mapping):
        raise ValueError("package_versions must be an object")
    missing_packages = [
        name for name in RUNTIME_PACKAGE_NAMES if name not in package_versions
    ]
    if missing_packages:
        raise ValueError(
            "package_versions is missing: " + ", ".join(missing_packages)
        )

    if manifest.get("git_dirty") not in (True, False, None):
        raise ValueError("git_dirty must be boolean or null")

    config_sha256 = manifest.get("config_sha256")
    if config_sha256 is not None:
        if (
            not isinstance(config_sha256, str)
            or len(config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in config_sha256)
        ):
            raise ValueError("config_sha256 must be a lowercase SHA-256 digest")

    jax_info = manifest["jax"]
    if not isinstance(jax_info, Mapping):
        raise ValueError("jax must be an object")
    if not isinstance(jax_info.get("available"), bool):
        raise ValueError("jax.available must be boolean")
    if not isinstance(jax_info.get("devices"), list):
        raise ValueError("jax.devices must be an array")


def write_runtime_manifest(
    manifest: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Validate and write a canonical manifest without overwriting a file."""

    validate_runtime_manifest(manifest)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            manifest,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    return output
