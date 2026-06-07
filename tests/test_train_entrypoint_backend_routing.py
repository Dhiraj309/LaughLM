import pytest

from LaughLM.config.loader import load_config
from scripts.train import (
    resolve_backend,
    resolve_trainer_class,
    resolve_data_replicas,
    Trainer,
    FSDPTrainer,
)


def test_train_entrypoint_resolves_pmap_trainer():
    cfg = load_config("configs/v5e_pmap.yaml")

    assert resolve_backend(cfg) == "pmap"
    assert resolve_trainer_class(cfg) is Trainer


def test_train_entrypoint_resolves_fsdp_trainer():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    assert resolve_backend(cfg) == "fsdp"
    assert resolve_trainer_class(cfg) is FSDPTrainer


def test_train_entrypoint_resolves_gspmd_alias_to_fsdp():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    cfg.runtime.backend = "gspmd"

    assert resolve_backend(cfg) == "fsdp"
    assert resolve_trainer_class(cfg) is FSDPTrainer


@pytest.mark.parametrize("backend", ["parallel3d", "moe"])
def test_train_entrypoint_rejects_reserved_backends(backend):
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    cfg.runtime.backend = backend

    with pytest.raises(
        NotImplementedError,
        match=backend,
    ):
        resolve_trainer_class(cfg)


@pytest.mark.parametrize("backend", ["parallel3d", "moe"])
def test_train_entrypoint_rejects_reserved_backend_data_replicas(backend):
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    cfg.runtime.backend = backend

    with pytest.raises(
        NotImplementedError,
        match=backend,
    ):
        resolve_data_replicas(cfg)


def test_train_entrypoint_rejects_unknown_backend_data_replicas():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    # Simulate a corrupted/unknown backend after schema validation.
    cfg.runtime.backend = "unknown_backend"

    with pytest.raises(
        ValueError,
        match="Cannot resolve data replicas",
    ):
        resolve_data_replicas(cfg)
