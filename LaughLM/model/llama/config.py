"""
LaughLM/model/llama/config.py

Canonical Llama-family architecture configuration.

Design goals:
- HF-compatible semantics
- deterministic architecture invariants
- minimal architectural surface
- no transformer-zoo polymorphism
- future-compatible with:
    - HF checkpoint conversion
    - PEFT / LoRA / QLoRA
    - tensor parallelism
    - GSPMD sharding

This config intentionally models only:
- decoder-only
- text-only
- Llama-style architectures

It does NOT contain:
- training config
- optimizer config
- runtime config
- distributed config
- kernel dispatch config
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    """
    Minimal canonical Llama architecture config.

    Tensor conventions
    ------------------
    hidden_states:
        [batch, seq, hidden_size]

    attention q/k/v:
        [batch, seq, num_heads, head_dim]

    KV cache:
        [batch, cache_seq, num_key_value_heads, head_dim]
    """

    # ──────────────────────────────────────────────────────────
    # Vocabulary
    # ──────────────────────────────────────────────────────────

    vocab_size: int = 32000

    # ──────────────────────────────────────────────────────────
    # Core architecture
    # ──────────────────────────────────────────────────────────

    hidden_size: int = 4096

    intermediate_size: int = 11008

    num_hidden_layers: int = 32

    num_attention_heads: int = 32

    num_key_value_heads: Optional[int] = None

    head_dim: Optional[int] = None

    max_position_embeddings: int = 2048

    # ──────────────────────────────────────────────────────────
    # Attention / RoPE
    # ──────────────────────────────────────────────────────────

    rope_theta: float = 10000.0

    rope_scaling: Optional[float] = None

    attention_bias: bool = False

    attention_dropout: float = 0.0

    # ──────────────────────────────────────────────────────────
    # MLP / normalization
    # ──────────────────────────────────────────────────────────

    hidden_act: str = "silu"

    rms_norm_eps: float = 1e-6

    mlp_bias: bool = False

    # ──────────────────────────────────────────────────────────
    # Embeddings / logits
    # ──────────────────────────────────────────────────────────

    tie_word_embeddings: bool = False

    # ──────────────────────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────────────────────

    initializer_range: float = 0.02

    # ──────────────────────────────────────────────────────────
    # Tokens
    # ──────────────────────────────────────────────────────────

    pad_token_id: Optional[int] = None

    bos_token_id: int = 1

    eos_token_id: int = 2

    # ──────────────────────────────────────────────────────────
    # Cache
    # ──────────────────────────────────────────────────────────

    use_cache: bool = True

    # ──────────────────────────────────────────────────────────
    # Validation / derived fields
    # ──────────────────────────────────────────────────────────

    def __post_init__(self):

        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = (
                self.hidden_size // self.num_attention_heads
            )

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by "
                "num_attention_heads"
            )

        if (
            self.num_attention_heads
            % self.num_key_value_heads
            != 0
        ):
            raise ValueError(
                "num_attention_heads must be divisible by "
                "num_key_value_heads for GQA"
            )

        expected_hidden = (
            self.num_attention_heads * self.head_dim
        )

        if expected_hidden != self.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: "
                f"{expected_hidden=} != "
                f"{self.hidden_size=}"
            )