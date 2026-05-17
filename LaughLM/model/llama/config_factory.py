"""
LaughLM/model/llama/config_factory.py

Factory utilities for constructing canonical
LlamaConfig objects from experiment configs.

Design goals
------------
- single source of truth
- deterministic architecture mapping
- HF-compatible Llama semantics
- TPU-safe derived dimensions
"""

from LaughLM.config.schema import (
    LaughLMConfig,
)

from LaughLM.model.llama.config import (
    LlamaConfig,
)


def build_llama_config(
    config: LaughLMConfig,
) -> LlamaConfig:
    """
    Convert experiment config into canonical
    Llama architecture config.
    """

    model = config.model
    arch = config.architecture
    init = config.initialization

    # --------------------------------------------------
    # Intermediate size
    #
    # Canonical SwiGLU:
    #
    # intermediate = round_up(
    #     8 * hidden_size / 3
    # )
    #
    # Llama-style uses ~2.666x hidden dim.
    # --------------------------------------------------

    intermediate_size = int(
        (8 * model.d_model) / 3
    )

    #
    # TPU alignment
    #
    # Round to multiple of 256 for MXU efficiency.
    #

    intermediate_size = (
        (intermediate_size + 255) // 256
    ) * 256

    return LlamaConfig(
        # Vocabulary
        vocab_size=model.vocab_size,

        # Core architecture
        hidden_size=model.d_model,
        intermediate_size=intermediate_size,
        num_hidden_layers=model.num_layers,
        num_attention_heads=model.num_heads,
        num_key_value_heads=model.num_kv_heads,
        max_position_embeddings=model.max_seq_len,

        # Attention
        rope_theta=10000.0,
        attention_bias=arch.bias,

        # MLP / norm
        hidden_act="silu",
        rms_norm_eps=1e-6,
        mlp_bias=arch.bias,

        # Embeddings
        tie_word_embeddings=arch.weight_tying,

        # Initialization
        initializer_range=init.std,
    )
