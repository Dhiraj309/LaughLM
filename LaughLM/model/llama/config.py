"""
LaughLM/model/llama/config.py

Canonical Llama-family architecture configuration.
"""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp


@dataclass
class LlamaConfig:

    # =========================================================
    # Vocabulary
    # =========================================================

    vocab_size: int = 32000

    # =========================================================
    # Core architecture
    # =========================================================

    hidden_size: int = 4096

    intermediate_size: int = 11008

    num_hidden_layers: int = 32

    num_attention_heads: int = 32

    num_key_value_heads: Optional[int] = None

    head_dim: Optional[int] = None

    max_position_embeddings: int = 2048

    # =========================================================
    # Attention / RoPE
    # =========================================================

    rope_theta: float = 10000.0

    rope_scaling: Optional[float] = None

    attention_bias: bool = False

    attention_dropout: float = 0.0

    # PMAP production attention backend:
    # - "standard" / "xla": JAX XLA dot_product_attention
    # - "flash" / "cudnn" / "memory_efficient": currently routed to XLA SDPA
    # - "splash": TPU SplashAttention when eligible
    attention_impl: str = "standard"

    # Splash fallback policy:
    # - "warn": warn and fall back to XLA SDPA
    # - "error": raise immediately, useful for benchmark configs
    attention_fallback: str = "warn"
    
    fused_qkv: bool = False

    # =========================================================
    # MLP / normalization
    # =========================================================

    hidden_act: str = "silu"

    rms_norm_eps: float = 1e-6

    mlp_bias: bool = False

    # =========================================================
    # Architecture
    # =========================================================

    parallel_block: bool = False

    # =========================================================
    # Embeddings / logits
    # =========================================================

    tie_word_embeddings: bool = False

    # =========================================================
    # Initialization
    # =========================================================

    initializer_range: float = 0.02

    # =========================================================
    # Tokens
    # =========================================================

    pad_token_id: Optional[int] = None

    bos_token_id: int = 1

    eos_token_id: int = 2

    # =========================================================
    # Cache
    # =========================================================

    use_cache: bool = True

    # =========================================================
    # DTypes
    # =========================================================

    param_dtype: jnp.dtype = jnp.float32

    compute_dtype: jnp.dtype = jnp.bfloat16

    output_dtype: jnp.dtype = jnp.float32
    
    optimizations: Optional[object] = None


    # =========================================================
    # Validation
    # =========================================================

    def __post_init__(self):

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")

        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be > 0")

        if self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be > 0")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        expected_hidden = self.num_attention_heads * self.head_dim

        if expected_hidden != self.hidden_size:
            raise ValueError(
                f"Inconsistent dimensions:\n"
                f"  hidden_size={self.hidden_size}\n"
                f"  num_attention_heads={self.num_attention_heads}\n"
                f"  head_dim={self.head_dim}"
            )

        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")

        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be > 0")

        valid_attention_impls = {
            "standard",
            "xla",
            "flash",
            "cudnn",
            "memory_efficient",
            "splash",
        }

        if self.attention_impl not in valid_attention_impls:
            raise ValueError(
                f"Unknown attention_impl: {self.attention_impl!r}. "
                f"Valid values: {sorted(valid_attention_impls)}"
            )

        valid_attention_fallbacks = {
            "warn",
            "error",
        }

        if self.attention_fallback not in valid_attention_fallbacks:
            raise ValueError(
                f"Unknown attention_fallback: {self.attention_fallback!r}. "
                f"Valid values: {sorted(valid_attention_fallbacks)}"
            )