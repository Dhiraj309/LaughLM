from LaughLM.config.loader import load_config


def test_pmap_config_backend():
    cfg = load_config("configs/v5e_pmap.yaml")

    assert cfg.runtime.backend == "pmap"

    axis_sizes = cfg.spmd.mesh.axis_sizes()

    assert axis_sizes["data"] == 8
    assert axis_sizes["fsdp"] == 1
    assert axis_sizes["tensor"] == 1
    assert axis_sizes["sequence"] == 1
    assert axis_sizes["pipeline"] == 1

    assert cfg.spmd.mesh.total_devices() == 8


def test_fsdp_config_backend():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    # Current compatibility name.
    # Phase 2 will introduce canonical "fsdp" with "gspmd" alias.
    assert cfg.runtime.backend == "gspmd"

    axis_sizes = cfg.spmd.mesh.axis_sizes()

    assert axis_sizes["data"] == 2
    assert axis_sizes["fsdp"] == 4
    assert axis_sizes["tensor"] == 1
    assert axis_sizes["sequence"] == 1
    assert axis_sizes["pipeline"] == 1

    assert cfg.spmd.mesh.total_devices() == 8
