"""
LaughLM/export/hf_config.py
"""

from __future__ import annotations

import json
from pathlib import Path


def build_hf_config(config):
    """
    Convert LaughLM config -> HF Llama config dict.
    """

    return {

        # --------------------------------------------------
        # Core architecture
        # --------------------------------------------------

        "architectures": [
            "LlamaForCausalLM",
        ],

        "model_type": "llama",

        "hidden_size":
            config.hidden_size,

        "intermediate_size":
            config.intermediate_size,

        "num_hidden_layers":
            config.num_hidden_layers,

        "num_attention_heads":
            config.num_attention_heads,

        "num_key_value_heads":
            config.num_key_value_heads,

        "max_position_embeddings":
            config.max_position_embeddings,

        # --------------------------------------------------
        # Vocabulary
        # --------------------------------------------------

        "vocab_size":
            config.vocab_size,

        # --------------------------------------------------
        # RoPE
        # --------------------------------------------------

        "rope_theta":
            config.rope_theta,

        # --------------------------------------------------
        # Norms
        # --------------------------------------------------

        "rms_norm_eps":
            config.rms_norm_eps,

        # --------------------------------------------------
        # Activations
        # --------------------------------------------------

        "hidden_act":
            config.hidden_act,

        # --------------------------------------------------
        # Attention
        # --------------------------------------------------

        "attention_bias":
            config.attention_bias,

        "attention_dropout":
            config.attention_dropout,

        # --------------------------------------------------
        # Embeddings
        # --------------------------------------------------

        "tie_word_embeddings":
            config.tie_word_embeddings,

        # --------------------------------------------------
        # Initialization
        # --------------------------------------------------

        "initializer_range":
            config.initializer_range,

        # --------------------------------------------------
        # Tokens
        # --------------------------------------------------

        "bos_token_id":
            config.bos_token_id,

        "eos_token_id":
            config.eos_token_id,

        "pad_token_id":
            config.pad_token_id,

        # --------------------------------------------------
        # DTypes
        # --------------------------------------------------

        "torch_dtype":
            _dtype_to_string(
                config.compute_dtype
            ),

        # --------------------------------------------------
        # HF compatibility
        # --------------------------------------------------

        "transformers_version":
            "4.57.0",

        "use_cache":
            config.use_cache,
    }


def build_generation_config(config):

    return {

        "bos_token_id":
            config.bos_token_id,

        "eos_token_id":
            config.eos_token_id,

        "pad_token_id":
            config.pad_token_id,

        "do_sample": True,

        "temperature": 1.0,

        "top_p": 1.0,

        "top_k": 50,
    }


def _dtype_to_string(dtype):

    s = str(dtype)

    if "bfloat16" in s:
        return "bfloat16"

    if "float16" in s:
        return "float16"

    return "float32"


def write_hf_configs(
    output_dir,
    config,
):
    """
    Write HF-compatible config files.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_json = build_hf_config(
        config
    )

    generation_json = (
        build_generation_config(
            config
        )
    )

    with open(
        output_dir / "config.json",
        "w",
    ) as f:

        json.dump(
            config_json,
            f,
            indent=2,
        )

    with open(
        output_dir
        / "generation_config.json",
        "w",
    ) as f:

        json.dump(
            generation_json,
            f,
            indent=2,
        )
