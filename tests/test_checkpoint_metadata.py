import copy

import pytest

from LaughLM.config.loader import load_config
from LaughLM.training.checkpoint import CheckpointManager


def test_checkpoint_metadata_v3_pmap_layout():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=7,
        tokens_processed=123_456,
        num_devices=8,
    )

    assert metadata["format"] == "laughlm_checkpoint_v3"
    assert metadata["backend"] == "pmap"
    assert metadata["raw_backend"] == "pmap"

    assert metadata["step"] == 7
    assert metadata["tokens_processed"] == 123_456

    assert metadata["runtime"]["backend"] == "pmap"
    assert metadata["runtime"]["canonical_backend"] == "pmap"

    assert metadata["layout"]["mesh_axes"] == [
        "data",
        "fsdp",
        "tensor",
        "sequence",
        "pipeline",
    ]

    assert metadata["layout"]["active_mesh_axes"] == [
        "data",
    ]

    assert metadata["layout"]["axis_sizes"] == {
        "data": 8,
        "fsdp": 1,
        "tensor": 1,
        "sequence": 1,
        "pipeline": 1,
    }

    assert metadata["layout"]["logical_axis_rules"]["batch"] == "data"

    assert metadata["architecture"]["fused_qkv"] is True
    assert metadata["architecture"]["weight_tying"] is True

    expected_tokens_per_step = (
        cfg.runtime.seq_len
        * cfg.runtime.micro_batch_per_device
        * 8
        * cfg.runtime.gradient_accumulation
    )

    assert metadata["tokens_per_step"] == expected_tokens_per_step


def test_checkpoint_metadata_v3_fsdp_gspmd_alias_layout():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=3,
        tokens_processed=98_304,
        num_devices=2,
    )

    assert metadata["format"] == "laughlm_checkpoint_v3"

    assert metadata["raw_backend"] == "gspmd"
    assert metadata["backend"] == "fsdp"

    assert metadata["runtime"]["backend"] == "gspmd"
    assert metadata["runtime"]["canonical_backend"] == "fsdp"

    assert metadata["layout"]["active_mesh_axes"] == [
        "data",
        "fsdp",
    ]

    assert metadata["layout"]["axis_sizes"] == {
        "data": 2,
        "fsdp": 4,
        "tensor": 1,
        "sequence": 1,
        "pipeline": 1,
    }

    assert metadata["layout"]["logical_axis_rules"]["batch"] == "data"
    assert metadata["layout"]["logical_axis_rules"]["embed"] == "fsdp"

    expected_tokens_per_step = (
        cfg.runtime.seq_len
        * cfg.runtime.micro_batch_per_device
        * 2
        * cfg.runtime.gradient_accumulation
    )

    assert metadata["tokens_per_step"] == expected_tokens_per_step


def test_checkpoint_metadata_preserves_legacy_validation_blocks():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    for key in (
        "model",
        "runtime",
        "optimizer",
        "scheduler",
        "parallelism",
        "architecture",
    ):
        assert key in metadata

    assert metadata["model"]["vocab_size"] == cfg.model.vocab_size
    assert metadata["model"]["d_model"] == cfg.model.d_model
    assert metadata["model"]["num_layers"] == cfg.model.num_layers

    assert metadata["optimizer"]["type"] == cfg.optimizer.type
    assert metadata["scheduler"]["type"] == cfg.scheduler.type

    assert (
        metadata["parallelism"]["data_parallel"]
        == cfg.parallelism.data_parallel
    )


def test_checkpoint_metadata_validation_accepts_matching_pmap_v3():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=8,
    )


def test_checkpoint_metadata_validation_accepts_matching_fsdp_v3():
    cfg = load_config("configs/v5e_fsdp_smoke.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=2,
    )

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=2,
    )


def test_checkpoint_metadata_validation_rejects_backend_mismatch():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    metadata = copy.deepcopy(metadata)
    metadata["backend"] = "fsdp"

    with pytest.raises(
        ValueError,
        match="backend/layout",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
        )


def test_checkpoint_metadata_validation_rejects_axis_size_mismatch():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    metadata = copy.deepcopy(metadata)
    metadata["layout"]["axis_sizes"]["fsdp"] = 2

    with pytest.raises(
        ValueError,
        match="layout.axis_sizes",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
        )


def test_checkpoint_metadata_validation_rejects_logical_axis_rule_mismatch():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    metadata = copy.deepcopy(metadata)
    metadata["layout"]["logical_axis_rules"]["embed"] = "fsdp"

    with pytest.raises(
        ValueError,
        match="layout.logical_axis_rules",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
        )


def test_checkpoint_metadata_validation_accepts_legacy_pmap_v2_without_layout():
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


def test_checkpoint_metadata_validation_rejects_unknown_format():
    cfg = load_config("configs/v5e_pmap.yaml")

    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=1,
        tokens_processed=1,
        num_devices=8,
    )

    metadata = copy.deepcopy(metadata)
    metadata["format"] = "unknown_format"

    with pytest.raises(
        ValueError,
        match="Unknown checkpoint metadata format",
    ):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=cfg,
            num_devices=8,
        )
