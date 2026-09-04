import copy

import pytest

from LaughLM.config.loader import load_config
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.scheduler import build_scheduler


CONFIG_PATH = "configs/production/laughlm_v1_127m_24b_extension.yaml"
TOKENS_PER_STEP = 2048 * 2 * 8 * 16
START_STEP = 20_000_000_000 // TOKENS_PER_STEP
END_STEP = 24_000_000_000 // TOKENS_PER_STEP


def _parent_metadata(config):
    metadata = CheckpointManager.build_metadata_from_config(
        config=config,
        step=START_STEP,
        tokens_processed=START_STEP * TOKENS_PER_STEP,
        num_devices=8,
        state_token_counter_dtype="host-int64",
    )
    metadata = copy.deepcopy(metadata)
    metadata["runtime"]["total_tokens"] = 20_000_000_000
    metadata["runtime"]["total_steps"] = START_STEP
    metadata["scheduler"].update(
        {
            "type": "wsd",
            "horizon_tokens": 20_000_000_000,
            "total_steps": START_STEP,
            "warmup_steps": 381,
            "warmup_fraction": 0.01,
            "stable_fraction": 0.95,
            "decay_steps": 1_527,
            "min_lr_ratio": 0.05,
        }
    )
    return metadata


def test_extension_config_loads_with_reserved_data_split():
    config = load_config(CONFIG_PATH)

    assert config.scheduler.type == "continuation_decay"
    assert config.runtime.resume_mode == "scheduler_fork"
    assert config.runtime.total_tokens == 24_000_000_000
    assert config.data.train_shard_start == 80
    assert config.data.train_shard_count == 16
    assert config.data.validation_shard_start == 96
    assert config.data.validation_shard_count == 4


def test_continuation_schedule_is_continuous_and_monotonic():
    config = load_config(CONFIG_PATH)
    schedule = build_scheduler(config, num_devices=8)

    assert float(schedule(START_STEP)) == pytest.approx(1.0e-5)
    assert float(schedule(END_STEP)) == pytest.approx(1.0e-6)
    assert float(schedule(START_STEP + 1)) < float(schedule(START_STEP))
    assert float(schedule(END_STEP - 1)) > float(schedule(END_STEP))


def test_scheduler_fork_accepts_only_compatible_wsd_parent():
    config = load_config(CONFIG_PATH)
    metadata = _parent_metadata(config)

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=config,
        num_devices=8,
        require_metadata=True,
        require_v3=True,
        purpose="pmap_scheduler_fork",
    )


def test_normal_resume_still_rejects_scheduler_transition():
    config = load_config(CONFIG_PATH)
    metadata = _parent_metadata(config)

    with pytest.raises(ValueError, match="Checkpoint LR schedule"):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=config,
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="pmap_resume",
        )


def test_scheduler_fork_rejects_wrong_parent_step():
    config = load_config(CONFIG_PATH)
    metadata = _parent_metadata(config)
    metadata["step"] = START_STEP - 1

    with pytest.raises(ValueError, match="parent step"):
        CheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=config,
            num_devices=8,
            require_metadata=True,
            require_v3=True,
            purpose="pmap_scheduler_fork",
        )
