import copy

import pytest

from LaughLM.config.loader import load_config
from LaughLM.training.checkpoint import CheckpointManager


def _metadata_for_pmap():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    return cfg, metadata


def test_export_metadata_validation_rejects_missing_metadata():
    cfg = load_config("configs/v5e_pmap.yaml")

    with pytest.raises(
        ValueError,
        match="Checkpoint metadata is required",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=None,
            config=cfg,
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="export",
        )


def test_export_metadata_validation_rejects_legacy_v2_metadata():
    cfg, metadata = _metadata_for_pmap()

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
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="export",
        )


def test_resume_metadata_validation_still_accepts_legacy_v2_metadata():
    cfg, metadata = _metadata_for_pmap()

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


def test_export_metadata_validation_rejects_missing_dtype_policy():
    cfg, metadata = _metadata_for_pmap()

    metadata = copy.deepcopy(metadata)
    metadata.pop("dtype_policy", None)

    with pytest.raises(
        ValueError,
        match="dtype_policy",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="export",
        )


def test_export_metadata_validation_rejects_dtype_policy_mismatch():
    cfg, metadata = _metadata_for_pmap()

    metadata = copy.deepcopy(metadata)
    metadata["dtype_policy"]["spmd"]["param_dtype"] = "bfloat16"

    with pytest.raises(
        ValueError,
        match="dtype_policy",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="export",
        )


def test_export_metadata_validation_accepts_matching_v3_metadata():
    cfg, metadata = _metadata_for_pmap()

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=8,
        require_metadata=True,
        require_v3=True,
        purpose="export",
    )
