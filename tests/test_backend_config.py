from pydantic import ValidationError
import pytest

from LaughLM.config.loader import load_config
from LaughLM.config.schema import RuntimeConfig


def test_pmap_config_backend():
    cfg = load_config("configs/v5e_pmap.yaml")

    assert cfg.runtime.backend == "pmap"
    assert cfg.runtime.canonical_backend == "pmap"
    assert cfg.runtime.backend_is_alias is False

    axis_sizes = cfg.spmd.mesh.axis_sizes()

    assert axis_sizes["data"] == 8
    assert axis_sizes["fsdp"] == 1
    assert axis_sizes["tensor"] == 1
    assert axis_sizes["sequence"] == 1
    assert axis_sizes["pipeline"] == 1

    assert cfg.spmd.mesh.total_devices() == 8


def test_fsdp_config_backend_gspmd_alias():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    assert cfg.runtime.backend == "gspmd"
    assert cfg.runtime.canonical_backend == "fsdp"
    assert cfg.runtime.backend_is_alias is True

    axis_sizes = cfg.spmd.mesh.axis_sizes()

    assert axis_sizes["data"] == 2
    assert axis_sizes["fsdp"] == 4
    assert axis_sizes["tensor"] == 1
    assert axis_sizes["sequence"] == 1
    assert axis_sizes["pipeline"] == 1

    assert cfg.spmd.mesh.total_devices() == 8


def test_runtime_config_accepts_canonical_fsdp():
    runtime = RuntimeConfig(
        backend="fsdp",
        seq_len=1024,
        micro_batch_per_device=4,
        gradient_accumulation=1,
        total_tokens=1_000_000,
        eval_interval=100,
        log_interval=10,
        checkpoint_interval=100,
        checkpoint_max_to_keep=2,
        checkpoint_dir="checkpoints/test",
    )

    assert runtime.backend == "fsdp"
    assert runtime.canonical_backend == "fsdp"
    assert runtime.backend_is_alias is False


def test_runtime_config_accepts_reserved_backends():
    for backend in ("parallel3d", "moe"):
        runtime = RuntimeConfig(
            backend=backend,
            seq_len=1024,
            micro_batch_per_device=4,
            gradient_accumulation=1,
            total_tokens=1_000_000,
            eval_interval=100,
            log_interval=10,
            checkpoint_interval=100,
            checkpoint_max_to_keep=2,
            checkpoint_dir="checkpoints/test",
        )

        assert runtime.backend == backend
        assert runtime.canonical_backend == backend
        assert runtime.backend_is_alias is False


def test_runtime_config_rejects_invalid_backend():
    with pytest.raises(ValidationError):
        RuntimeConfig(
            backend="invalid",
            seq_len=1024,
            micro_batch_per_device=4,
            gradient_accumulation=1,
            total_tokens=1_000_000,
            eval_interval=100,
            log_interval=10,
            checkpoint_interval=100,
            checkpoint_max_to_keep=2,
            checkpoint_dir="checkpoints/test",
        )
