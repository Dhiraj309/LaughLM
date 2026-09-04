import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from LaughLM.provenance.environment import (
    RUNTIME_MANIFEST_SCHEMA,
    RUNTIME_PACKAGE_NAMES,
    build_runtime_manifest,
    validate_runtime_manifest,
    write_runtime_manifest,
)


class _FakeDevice:
    def __init__(self, device_id: int, kind: str):
        self.id = device_id
        self.platform = "cpu"
        self.device_kind = kind
        self.process_index = 0


class _FakeJax:
    config = SimpleNamespace(x64_enabled=False)

    @staticmethod
    def default_backend():
        return "cpu"

    @staticmethod
    def process_index():
        return 0

    @staticmethod
    def process_count():
        return 1

    @staticmethod
    def devices():
        return [_FakeDevice(0, "cpu"), _FakeDevice(1, "cpu")]


def _resolver(package_name: str) -> str:
    return f"test-{package_name}"


def test_runtime_manifest_is_versioned_and_deterministic(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_bytes = b"model:\n  d_model: 32\n"
    config_path.write_bytes(config_bytes)
    expected_digest = hashlib.sha256(config_bytes).hexdigest()

    kwargs = {
        "repo_root": tmp_path,
        "config_path": config_path,
        "captured_at_utc": "2026-09-04T00:00:00Z",
        "package_version_resolver": _resolver,
        "git_state": ("abc123", True),
        "jax_module": _FakeJax,
    }
    first = build_runtime_manifest(**kwargs)
    second = build_runtime_manifest(**kwargs)

    assert first == second
    assert first["manifest_schema"] == RUNTIME_MANIFEST_SCHEMA
    assert first["manifest_version"] == 1
    assert first["package_versions"] == {
        name: f"test-{name}" for name in RUNTIME_PACKAGE_NAMES
    }
    assert first["git_revision"] == "abc123"
    assert first["git_dirty"] is True
    assert first["config_sha256"] == expected_digest
    assert first["jax"]["backend"] == "cpu"
    assert first["jax"]["process_count"] == 1
    assert [device["id"] for device in first["jax"]["devices"]] == [0, 1]
    validate_runtime_manifest(first)


def test_runtime_manifest_writer_is_canonical_and_non_overwriting(tmp_path):
    manifest = build_runtime_manifest(
        repo_root=tmp_path,
        captured_at_utc="2026-09-04T00:00:00Z",
        package_version_resolver=_resolver,
        git_state=(None, None),
        jax_module=_FakeJax,
    )
    output = tmp_path / "artifacts" / "environment.json"

    assert write_runtime_manifest(manifest, output) == output.resolve()
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError):
        write_runtime_manifest(manifest, output)


def test_runtime_manifest_rejects_missing_package_version():
    manifest = build_runtime_manifest(
        package_version_resolver=_resolver,
        git_state=(None, None),
        jax_module=_FakeJax,
    )
    manifest["package_versions"].pop("tokamax")

    with pytest.raises(ValueError, match="tokamax"):
        validate_runtime_manifest(manifest)


def test_runtime_manifest_schema_file_is_versioned():
    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / "runtime_manifest_v1.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$id"].endswith("runtime_manifest_v1.json")
    assert schema["properties"]["manifest_schema"]["const"] == RUNTIME_MANIFEST_SCHEMA
    assert schema["properties"]["manifest_version"]["const"] == 1
