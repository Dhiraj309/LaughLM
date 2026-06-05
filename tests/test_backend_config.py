from pydantic import ValidationError
import pytest

from LaughLM.config.loader import load_config
from LaughLM.config.schema import RuntimeConfig
from LaughLM.config.validation import validate_config
from LaughLM.distributed.sharding import get_logical_axis_rules


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

    assert cfg.parallelism.data_parallel == axis_sizes["data"]
    assert cfg.parallelism.model_parallel == (
        axis_sizes["fsdp"]
        * axis_sizes["tensor"]
    )

    assert cfg.spmd.mesh.total_devices() == 8


def test_fsdp_config_backend_and_mesh_alignment():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    assert cfg.runtime.canonical_backend == "fsdp"

    axis_sizes = cfg.spmd.mesh.axis_sizes()

    assert axis_sizes["data"] >= 1
    assert axis_sizes["fsdp"] > 1
    assert axis_sizes["tensor"] == 1
    assert axis_sizes["sequence"] == 1
    assert axis_sizes["pipeline"] == 1

    assert cfg.parallelism.data_parallel == axis_sizes["data"]
    assert cfg.parallelism.model_parallel == (
        axis_sizes["fsdp"]
        * axis_sizes["tensor"]
    )

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


def test_config_validation_rejects_data_parallel_mesh_mismatch():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    cfg.parallelism.data_parallel = (
        cfg.spmd.mesh.axis_sizes()["data"]
        + 1
    )

    with pytest.raises(
        ValueError,
        match="parallelism.data_parallel",
    ):
        validate_config(cfg)


def test_config_validation_rejects_model_parallel_mesh_mismatch():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    cfg.parallelism.model_parallel = (
        cfg.parallelism.model_parallel
        + 1
    )

    with pytest.raises(
        ValueError,
        match="parallelism.model_parallel",
    ):
        validate_config(cfg)


def test_config_validation_rejects_pmap_with_fsdp_axis():
    cfg = load_config("configs/v5e_pmap.yaml")

    cfg.spmd.mesh.ici_fsdp_parallelism = 2

    with pytest.raises(
        ValueError,
        match="pure data-parallel mesh",
    ):
        validate_config(cfg)

def test_config_validation_rejects_fsdp_splash_without_active_data_axis():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    cfg.runtime.backend = "fsdp"
    cfg.spmd.mesh.ici_data_parallelism = 1
    cfg.spmd.mesh.ici_fsdp_parallelism = 8

    cfg.parallelism.data_parallel = 1
    cfg.parallelism.model_parallel = 8

    cfg.architecture.attention_impl = "splash"

    with pytest.raises(
        ValueError,
        match="active 'data' mesh axis",
    ):
        validate_config(cfg)


def test_logical_axis_rules_drop_inactive_mesh_axes():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    cfg.spmd.mesh.ici_data_parallelism = 1
    cfg.spmd.mesh.ici_fsdp_parallelism = 8
    cfg.spmd.axis_rules.batch = "data"
    cfg.spmd.axis_rules.embed = "fsdp"

    class DummyMesh:
        axis_names = ("fsdp",)

    rules = dict(
        get_logical_axis_rules(
            cfg,
            mesh=DummyMesh(),
        )
    )

    assert rules["batch"] is None
    assert rules["embed"] == "fsdp"
