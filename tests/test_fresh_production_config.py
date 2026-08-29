from LaughLM.config.loader import load_config
from LaughLM.training.checkpoint import CheckpointManager


CONFIG_PATH = "configs/production/laughlm_v1_135m_fresh_4b.yaml"


def test_fresh_production_config_is_standalone_and_exact():
    cfg = load_config(CONFIG_PATH)

    assert cfg.model.vocab_size == 32064
    assert cfg.model.d_model == 1024
    assert cfg.model.num_layers == 8
    assert cfg.model.num_heads == 8
    assert cfg.model.num_kv_heads == 4
    assert cfg.architecture.attention_variant == "gqa"
    assert cfg.architecture.weight_tying is True

    assert cfg.runtime.total_tokens == 4_000_000_000
    assert cfg.scheduler.horizon_tokens == 20_000_000_000
    assert cfg.runtime.checkpoint_dir.endswith(
        "laughlm_v1_135m_fresh_20b"
    )
    assert cfg.optimizations.async_checkpointing is False

    assert cfg.data.train_shard_start == 0
    assert cfg.data.train_shard_count == 16
    assert cfg.data.validation_shard_start == 16
    assert cfg.data.validation_shard_count == 2


def test_fresh_production_config_builds_export_safe_v3_metadata():
    cfg = load_config(CONFIG_PATH)
    metadata = CheckpointManager.build_metadata_from_config(
        config=cfg,
        step=3814,
        tokens_processed=3_999_268_864,
        num_devices=8,
        state_token_counter_dtype="host-int64",
    )

    CheckpointManager.validate_metadata_compatible(
        metadata=metadata,
        config=cfg,
        num_devices=8,
        require_metadata=True,
        require_v3=True,
        purpose="export",
    )
