import pytest

from LaughLM.config.loader import load_config
from LaughLM.export.export_hf import (
    _canonical_backend,
    _metadata_num_devices,
    _require_supported_export_backend,
)


def test_hf_export_allows_pmap_backend():
    cfg = load_config("configs/v5e_pmap.yaml")

    assert _canonical_backend(cfg) == "pmap"

    _require_supported_export_backend(cfg)


def test_hf_export_blocks_fsdp_until_canonical_unshard_exists():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    assert _canonical_backend(cfg) == "fsdp"

    with pytest.raises(
        NotImplementedError,
        match="canonical unshard",
    ):
        _require_supported_export_backend(cfg)


@pytest.mark.parametrize("backend", ["parallel3d", "moe"])
def test_hf_export_blocks_reserved_backends(backend):
    cfg = load_config("configs/v5e_pmap.yaml")
    cfg.runtime.backend = backend

    assert _canonical_backend(cfg) == backend

    with pytest.raises(
        NotImplementedError,
        match="Reserved backend",
    ):
        _require_supported_export_backend(cfg)


def test_hf_export_pmap_metadata_num_devices_is_positive():
    cfg = load_config("configs/v5e_pmap.yaml")

    num_devices = _metadata_num_devices(cfg)

    assert isinstance(num_devices, int)
    assert num_devices > 0


def test_hf_export_fsdp_metadata_num_devices_uses_data_axis():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    expected = cfg.spmd.mesh.axis_sizes()["data"]

    assert _metadata_num_devices(cfg) == expected
