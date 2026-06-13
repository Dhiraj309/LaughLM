from LaughLM.config.loader import load_config


def test_v5e_fsdp_1p3b_optimized_config_guardrails():
    cfg = load_config(
        "configs/v5e_fsdp_1p3b_d4_f2_optimized.yaml"
    )

    data_replicas = cfg.spmd.mesh.axis_sizes()["data"]

    tokens_per_step = (
        cfg.runtime.seq_len
        * cfg.runtime.micro_batch_per_device
        * data_replicas
        * cfg.runtime.gradient_accumulation
    )

    total_steps = (
        cfg.runtime.total_tokens
        // tokens_per_step
    )

    assert cfg.runtime.backend == "fsdp"

    assert cfg.model.d_model == 2048
    assert cfg.model.num_layers == 24
    assert cfg.model.num_heads == 32
    assert cfg.model.num_kv_heads == 32

    assert cfg.architecture.attention_impl == "splash"
    assert cfg.architecture.fused_qkv is True
    assert cfg.architecture.weight_tying is True

    assert cfg.runtime.seq_len == 1024
    assert cfg.runtime.micro_batch_per_device == 2
    assert cfg.runtime.gradient_accumulation == 16

    assert cfg.loss.chunked_logits is True
    assert cfg.loss.logits_chunk_size == 4096
    assert cfg.loss.remat_logits_chunks is False

    assert cfg.runtime.benchmark_mode is False
    assert cfg.runtime.metrics_interval == 0

    assert data_replicas == 4
    assert cfg.spmd.mesh.axis_sizes()["fsdp"] == 2

    assert tokens_per_step == 131_072
    assert total_steps == 763
