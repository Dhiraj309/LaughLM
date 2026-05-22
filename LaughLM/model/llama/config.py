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

    # =========================================================
    # Runtime attention
    # =========================================================

    attention_backend: str = "flash"

    attention_block_q: int = 128

    attention_block_kv: int = 128

    attention_mask_type: str = "causal"

    sliding_window: Optional[int] = None

    chunk_size: Optional[int] = None

    # =========================================================
    # Rematerialization
    # =========================================================

    remat_attention: bool = False

    remat_mlp: bool = False

    remat_policy: Optional[str] = None

    prevent_cse: bool = False

    # =========================================================
    # Architecture
    # =========================================================

    parallel_block: bool = False

    scan_layers: bool = False

    # =========================================================
    # MLP / normalization
    # =========================================================

    hidden_act: str = "silu"

    rms_norm_eps: float = 1e-6

    mlp_bias: bool = False

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

    # =========================================================
    # Validation
    # =========================================================

    def __post_init__(self):

        if self.num_key_value_heads is None:

            self.num_key_value_heads = (
                self.num_attention_heads
            )

        # -----------------------------------------------------
        # Core dims
        # -----------------------------------------------------

        if self.hidden_size <= 0:

            raise ValueError(
                "hidden_size must be > 0"
            )

        if self.intermediate_size <= 0:

            raise ValueError(
                "intermediate_size must be > 0"
            )

        if self.num_hidden_layers <= 0:

            raise ValueError(
                "num_hidden_layers must be > 0"
            )

        if self.num_attention_heads <= 0:

            raise ValueError(
                "num_attention_heads must be > 0"
            )

        if self.num_key_value_heads <= 0:

            raise ValueError(
                "num_key_value_heads must be > 0"
            )

        # -----------------------------------------------------
        # Head divisibility
        # -----------------------------------------------------

        if (
            self.hidden_size
            %
            self.num_attention_heads
            != 0
        ):

            raise ValueError(
                "hidden_size must be divisible "
                "by num_attention_heads"
            )

        if (
            self.num_attention_heads
            %
            self.num_key_value_heads
            != 0
        ):

            raise ValueError(
                "num_attention_heads must be divisible "
                "by num_key_value_heads"
            )

        # -----------------------------------------------------
        # Head dim
        # -----------------------------------------------------

        if self.head_dim is None:

            self.head_dim = (
                self.hidden_size
                //
                self.num_attention_heads
            )

        expected_hidden = (
            self.num_attention_heads
            * self.head_dim
        )

        if expected_hidden != self.hidden_size:

            raise ValueError(
                f"Inconsistent dimensions:\n"
                f"hidden_size={self.hidden_size}\n"
                f"num_attention_heads="
                f"{self.num_attention_heads}\n"
                f"head_dim={self.head_dim}"
            )

        # -----------------------------------------------------
        # Sequence limits
        # -----------------------------------------------------

        if self.max_position_embeddings <= 0:

            raise ValueError(
                "max_position_embeddings "
                "must be > 0"
            )

        # -----------------------------------------------------
        # Attention backend
        # -----------------------------------------------------

        valid_backends = {
            "reference",
            "online",
            "flash",
            "decode",
        }

        if (
            self.attention_backend
            not in valid_backends
        ):

            raise ValueError(
                f"Unknown attention backend: "
                f"{self.attention_backend}"
            )

        # -----------------------------------------------------
        # Mask type
        # -----------------------------------------------------

        valid_masks = {
            "causal",
            "full",
            "sliding_window",
            "chunked",
        }

        if (
            self.attention_mask_type
            not in valid_masks
        ):

            raise ValueError(
                f"Unknown attention mask type: "
                f"{self.attention_mask_type}"
            )

        # -----------------------------------------------------
        # Sliding window validation
        # -----------------------------------------------------

        if (
            self.attention_mask_type
            == "sliding_window"
        ):

            if self.sliding_window is None:

                raise ValueError(
                    "sliding_window mask requires "
                    "sliding_window to be set"
                )

            if self.sliding_window <= 0:

                raise ValueError(
                    "sliding_window must be > 0"
                )

        # -----------------------------------------------------
        # Chunked validation
        # -----------------------------------------------------

        if (
            self.attention_mask_type
            == "chunked"
        ):

            if self.chunk_size is None:

                raise ValueError(
                    "chunked mask requires "
                    "chunk_size to be set"
                )

            if self.chunk_size <= 0:

                raise ValueError(
                    "chunk_size must be > 0"
                )

        # -----------------------------------------------------
        # Block sizes
        # -----------------------------------------------------

        if self.attention_block_q <= 0:

            raise ValueError(
                "attention_block_q must be > 0"
            )

        if self.attention_block_kv <= 0:

            raise ValueError(
                "attention_block_kv must be > 0"
            )

        # -----------------------------------------------------
        # Remat validation
        # -----------------------------------------------------

        if (
            self.remat_attention
            or self.remat_mlp
        ):

            if self.remat_policy is None:

                raise ValueError(
                    "remat enabled but "
                    "remat_policy is None"
                )

        valid_remat_policies = {
            None,
            "nothing_saveable",
            "dots_saveable",
            "dots_with_no_batch_dims_saveable",
            "everything_saveable",
        }

        if (
            self.remat_policy
            not in valid_remat_policies
        ):

            raise ValueError(
                f"Unknown remat policy: "
                f"{self.remat_policy}"
            )
