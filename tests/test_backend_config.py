from LaughLM.config.loader import load_config


def test_pmap_config_backend():
    cfg = load_config("configs/v5e_pmap.yaml")
    assert cfg.runtime.backend == "pmap"


def test_fsdp_config_backend():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    assert cfg.runtime.backend == "gspmd"
    assert cfg.spmd.mesh.axis_sizes()["fsdp"] == 8