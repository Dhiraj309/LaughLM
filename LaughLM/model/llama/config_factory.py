"""
LaughLM/model/llama/config_factory.py
"""

import jax.numpy as jnp

from LaughLM.config.schema import (
    LaughLMConfig,
)

from LaughLM.model.llama.config import (
    LlamaConfig,
)


DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
}


def build_llama_config(
    config: LaughLMConfig,
) -> LlamaConfig:
    """
    Convert experiment config into canonical
    LlamaConfig.
    """

    model = config.model

    arch = config.architecture

    init = config.initialization

    dtype_cfg = config.spmd.dtype

    # =========================================================
    # SwiGLU intermediate dim
    # =========================================================

    intermediate_size = int(
        (8 * model.d_model) / 3
    )

    # TPU/GPU alignment

    intermediate_size = (
        (intermediate_size + 255)
        // 256
    ) * 256

    # =========================================================
    # DTypes
    # =========================================================

    param_dtype = DTYPE_MAP[
        dtype_cfg.param_dtype
    ]

    compute_dtype = DTYPE_MAP[
        dtype_cfg.compute_dtype
    ]

    output_dtype = DTYPE_MAP[
        dtype_cfg.output_dtype
    ]

    # =========================================================
    # Build config
    # =========================================================

    return LlamaConfig(

        # Vocabulary
        vocab_size=model.vocab_size,

        # Core architecture
        hidden_size=model.d_model,

        intermediate_size=intermediate_size,

        num_hidden_layers=model.num_layers,

        num_attention_heads=model.num_heads,

        num_key_value_heads=model.num_kv_heads,

        max_position_embeddings=(
            model.max_seq_len
        ),

        # Attention
        rope_theta=10000.0,

        attention_bias=arch.bias,

        attention_dropout=0.0,

        # MLP
        hidden_act="silu",

        rms_norm_eps=1e-6,

        mlp_bias=arch.bias,

        # Architecture
        parallel_block=(
            arch.parallel_block
        ),

        # Embeddings
        tie_word_embeddings=(
            arch.weight_tying
        ),

        # Initialization
        initializer_range=init.std,

        # DTypes
        param_dtype=param_dtype,

        compute_dtype=compute_dtype,

        output_dtype=output_dtype,
    )
