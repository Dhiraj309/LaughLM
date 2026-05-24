"""
LaughLM/model/llama/config_factory.py

Build canonical LlamaConfig from LaughLMConfig.
"""

import jax.numpy as jnp

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.llama.config import LlamaConfig
from LaughLM.model.parameter_utils import compute_llama_intermediate_size


DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def _dtype(name: str):
    try:
        return DTYPE_MAP[name]
    except KeyError as e:
        raise ValueError(
            f"Unsupported dtype '{name}'. "
            f"Expected one of {sorted(DTYPE_MAP)}."
        ) from e


def build_llama_config(config: LaughLMConfig) -> LlamaConfig:
    model = config.model
    arch = config.architecture
    init = config.initialization

    intermediate_size = compute_llama_intermediate_size(
        model.d_model,
        multiple_of=256,
    )

    param_dtype = _dtype(config.parallelism.param_dtype)
    compute_dtype = _dtype(config.parallelism.compute_dtype)
    output_dtype = jnp.float32

    return LlamaConfig(
        vocab_size=model.vocab_size,
        hidden_size=model.d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=model.num_layers,
        num_attention_heads=model.num_heads,
        num_key_value_heads=model.num_kv_heads,
        max_position_embeddings=model.max_seq_len,
        rope_theta=10000.0,
        attention_bias=arch.bias,
        attention_dropout=0.0,
        attention_impl=arch.attention_impl,
        attention_fallback=getattr(arch, "attention_fallback", "warn"),
        fused_qkv=getattr(arch, "fused_qkv", False),
        hidden_act="silu",
        rms_norm_eps=1e-6,
        mlp_bias=arch.bias,
        parallel_block=arch.parallel_block,
        tie_word_embeddings=arch.weight_tying,
        initializer_range=init.std,
        param_dtype=param_dtype,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )