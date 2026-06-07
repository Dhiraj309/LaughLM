import pytest

from LaughLM.config.loader import load_config
from LaughLM.training.metadata import build_checkpoint_metadata


class DummyMesh:
    axis_names = ("data",)

    class Devices:
        shape = (8,)

    devices = Devices()


def test_legacy_build_checkpoint_metadata_rejects_default_use():
    cfg = load_config("configs/v5e_pmap.yaml")

    with pytest.raises(
        RuntimeError,
        match="legacy",
    ):
        build_checkpoint_metadata(
            config=cfg,
            mesh=DummyMesh(),
            step=1,
            tokens_processed=1,
        )


def test_legacy_build_checkpoint_metadata_requires_explicit_opt_in():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = build_checkpoint_metadata(
        config=cfg,
        mesh=DummyMesh(),
        step=1,
        tokens_processed=1,
        allow_legacy_v2=True,
    )

    assert metadata["format"] == "laughlm_pmap_checkpoint_v2"
