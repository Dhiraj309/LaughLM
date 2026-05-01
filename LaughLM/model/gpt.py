"""
LaughLM/model/gpt.py

Top-level GPT model for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. Uses build_block() — correct remat wrapping via factory, not
   broken nn.remat(self._forward) pattern.

2. KV cache support — forward pass accepts and returns KV caches
   for autoregressive generation. Each layer gets its own cache.

3. Output dtype — logits are always computed in float32 for numerical
   stability (cross-entropy with bf16 logits causes NaN).

4. NTK-aware RoPE — when config.architecture.positional == "rope_scaled",
   applies NTK-aware context extension via scale_factor.

5. Dtype from SPMD config — reads compute_dtype from config.spmd.dtype.

References:
  MaxText: AI-Hypercomputer/maxtext → layers.py (Decoder class)
  LLaMA: Meta — GPT architecture with RoPE + SwiGLU + RMSNorm
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, List

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.transformer_block import build_block
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.positional import (
    build_positional_encoding,
    build_rope_tables,
)
from LaughLM.model.layers.attention import KVCache
from LaughLM.utils.dtype import resolve_compute_dtype


class GPTModel(nn.Module):
    config: LaughLMConfig

    def setup(self):
        cfg = self.config
        d_model = cfg.model.d_model
        vocab_size = cfg.model.vocab_size
        num_layers = cfg.model.num_layers
        pos_type = cfg.architecture.positional

        self._compute_dtype = resolve_compute_dtype(cfg)

        # ── Token embedding ───────────────────────────────────
        self.token_embedding = nn.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nn.initializers.normal(
                stddev=cfg.initialization.embedding_std
            ),
        )

        # ── Positional encoding (additive only) ──────────────
        self.positional = build_positional_encoding(cfg)

        # ── RoPE tables ───────────────────────────────────────
        self._use_rope = pos_type in ("rope", "rope_scaled")

        if self._use_rope:
            head_dim = d_model // cfg.model.num_heads

            # NTK-aware scaling for rope_scaled
            scale_factor = None
            if pos_type == "rope_scaled":
                # Default 4× context extension
                scale_factor = 4.0

            self._rope_sin, self._rope_cos = build_rope_tables(
                head_dim=head_dim,
                max_seq_len=cfg.model.max_seq_len,
                scale_factor=scale_factor,
            )
        else:
            self._rope_sin = None
            self._rope_cos = None

        # ── Transformer blocks (with correct remat) ──────────
        self.blocks = [
            build_block(cfg)
            for _ in range(num_layers)
        ]

        # ── Final norm ────────────────────────────────────────
        self.final_norm = build_normalization(cfg)

        # ── LM head (if not weight-tying) ─────────────────────
        if not cfg.architecture.weight_tying:
            self.lm_head = nn.Dense(
                vocab_size,
                use_bias=cfg.architecture.bias,
                kernel_init=nn.initializers.normal(
                    stddev=cfg.initialization.std
                ),
            )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        doc_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> Tuple[jnp.ndarray, Optional[List[KVCache]]]:
        """
        Forward pass.

        Parameters
        ----------
        input_ids  : (B, T) integer token IDs
        doc_ids    : (B, T) integer segment/document IDs (for cross-doc masking)
        kv_caches  : list of KVCache per layer (for inference), or None

        Returns
        -------
        logits     : (B, T, V) in float32
        new_caches : list of updated KVCache per layer, or None
        """
        assert input_ids.ndim == 2, f"Expected (B, T), got {input_ids.shape}"

        B, T = input_ids.shape

        # ── Token embedding ───────────────────────────────────
        x = self.token_embedding(input_ids)
        x = x.astype(self._compute_dtype)

        # ── Positional encoding (additive, for learned/sinusoidal) ──
        if self.positional is not None:
            positions = jnp.arange(T)[None, :]
            pos_emb = self.positional(positions)
            assert pos_emb.ndim == 3
            x = x + pos_emb.astype(self._compute_dtype)

        # ── RoPE tables (slice to current seq_len) ────────────
        rope_tables: Optional[Tuple] = None
        if self._use_rope:
            rope_tables = (
                self._rope_sin[:T],
                self._rope_cos[:T],
            )

        # ── Transformer stack ─────────────────────────────────
        new_caches = [] if kv_caches is not None else None

        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None

            x, new_cache = block(
                x,
                rope_tables=rope_tables,
                doc_ids=doc_ids,
                kv_cache=layer_cache,
            )

            if new_caches is not None:
                new_caches.append(new_cache)

        # ── Final norm ────────────────────────────────────────
        x = self.final_norm(x)

        # ── Logits in float32 (CRITICAL for numerical stability) ──
        x = x.astype(jnp.float32)

        if self.config.architecture.weight_tying:
            embedding_table = self.token_embedding.embedding
            logits = jnp.einsum("btd,vd->btv", x, embedding_table)
        else:
            logits = self.lm_head(x)

        return logits, new_caches
