import copy

import pytest

from LaughLM.config.loader import load_config
from LaughLM.training.checkpoint import CheckpointManager


def _fsdp_cfg_and_metadata():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    num_devices = cfg.spmd.mesh.axis_sizes()["data"]

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=num_devices,
    )

    return cfg, metadata, num_devices


def test_fsdp_resume_rejects_missing_metadata():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")
    num_devices = cfg.spmd.mesh.axis_sizes()["data"]

    with pytest.raises(
        ValueError,
        match="Checkpoint metadata is required",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=None,
            config=cfg,
            num_devices=num_devices,
            require_metadata=True,
            require_v3=True,
            purpose="fsdp_resume",
        )


def test_fsdp_resume_rejects_legacy_v2_metadata():
    cfg, metadata, num_devices = _fsdp_cfg_and_metadata()

    metadata = copy.deepcopy(metadata)
    metadata["format"] = "laughlm_pmap_checkpoint_v2"
    metadata.pop("backend", None)
    metadata.pop("raw_backend", None)
    metadata.pop("layout", None)
    metadata.pop("dtype_policy", None)
    metadata["runtime"].pop("canonical_backend", None)

    with pytest.raises(
        ValueError,
        match="not export-safe",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=num_devices,
            require_metadata=True,
            require_v3=True,
            purpose="fsdp_resume",
        )


def test_fsdp_resume_accepts_matching_v3_metadata():
    cfg, metadata, num_devices = _fsdp_cfg_and_metadata()

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=num_devices,
        require_metadata=True,
        require_v3=True,
        purpose="fsdp_resume",
    )


def test_pmap_resume_still_allows_legacy_v2_metadata_by_default():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    metadata = copy.deepcopy(metadata)
    metadata["format"] = "laughlm_pmap_checkpoint_v2"
    metadata.pop("backend", None)
    metadata.pop("raw_backend", None)
    metadata.pop("layout", None)
    metadata.pop("dtype_policy", None)
    metadata["runtime"].pop("canonical_backend", None)

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=8,
    )
