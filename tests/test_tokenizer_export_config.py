from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.export.hf_config import (
    build_hf_config,
    build_generation_config,
)


def test_llama_config_uses_phi35_base_lm_token_ids():
    cfg = load_config("configs/v5e_pmap.yaml")
    llama_config = build_llama_config(cfg)

    assert llama_config.bos_token_id == 1
    assert llama_config.eos_token_id == 32000
    assert llama_config.pad_token_id == 32000

    assert llama_config.eos_token_id != 2


def test_hf_config_preserves_vocab_size_and_special_ids():
    cfg = load_config("configs/v5e_pmap.yaml")
    llama_config = build_llama_config(cfg)

    hf_config = build_hf_config(llama_config)

    assert hf_config["model_type"] == "llama"
    assert hf_config["architectures"] == ["LlamaForCausalLM"]

    assert hf_config["vocab_size"] == cfg.model.vocab_size
    assert hf_config["vocab_size"] == llama_config.vocab_size

    assert hf_config["bos_token_id"] == 1
    assert hf_config["eos_token_id"] == 32000
    assert hf_config["pad_token_id"] == 32000

    assert hf_config["eos_token_id"] != 2


def test_hf_config_preserves_tied_embedding_flag():
    cfg = load_config("configs/v5e_pmap.yaml")
    llama_config = build_llama_config(cfg)

    hf_config = build_hf_config(llama_config)

    assert hf_config["tie_word_embeddings"] == cfg.architecture.weight_tying
    assert hf_config["tie_word_embeddings"] == llama_config.tie_word_embeddings


def test_generation_config_uses_single_base_lm_eos_token():
    cfg = load_config("configs/v5e_pmap.yaml")
    llama_config = build_llama_config(cfg)

    generation_config = build_generation_config(llama_config)

    assert generation_config["bos_token_id"] == 1
    assert generation_config["eos_token_id"] == 32000
    assert generation_config["pad_token_id"] == 32000

    assert generation_config["eos_token_id"] != 2
    assert not isinstance(generation_config["eos_token_id"], list)

    assert generation_config["do_sample"] is False


def test_hf_config_core_dimensions_match_llama_config():
    cfg = load_config("configs/v5e_pmap.yaml")
    llama_config = build_llama_config(cfg)

    hf_config = build_hf_config(llama_config)

    assert hf_config["hidden_size"] == llama_config.hidden_size
    assert hf_config["intermediate_size"] == llama_config.intermediate_size
    assert hf_config["num_hidden_layers"] == llama_config.num_hidden_layers
    assert hf_config["num_attention_heads"] == llama_config.num_attention_heads
    assert hf_config["num_key_value_heads"] == llama_config.num_key_value_heads
    assert hf_config["max_position_embeddings"] == llama_config.max_position_embeddings
    assert hf_config["rms_norm_eps"] == llama_config.rms_norm_eps
    assert hf_config["rope_theta"] == llama_config.rope_theta
