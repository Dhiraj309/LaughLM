
import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple

from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.transformer_block import TransformerBlock
from LaughLM.model.layers.normalization import build_normalization
from LaughLM.model.layers.positional import (
    build_positional_encoding,
    build_rope_tables,
)


class GPTModel(nn.Module):
    config: LaughLMConfig

    def setup(self):

        cfg         = self.config
        d_model     = cfg.model.d_model
        vocab_size  = cfg.model.vocab_size
        num_layers  = cfg.model.num_layers
        pos_type    = cfg.architecture.positional
        compute_bf16 = (cfg.parallelism.compute_dtype == "bfloat16")

        self._compute_dtype = jnp.bfloat16 if compute_bf16 else jnp.float32

        # ------------------------------------------------------------
        # Token embedding
        # ------------------------------------------------------------
        self.token_embedding = nn.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nn.initializers.normal(
                stddev=cfg.initialization.embedding_std
            ),
        )

        # ------------------------------------------------------------
        # Positional encoding (additive only)
        # ------------------------------------------------------------
        self.positional = build_positional_encoding(cfg)

        # ------------------------------------------------------------
        # RoPE
        # ------------------------------------------------------------
        self._use_rope = pos_type in ("rope", "rope_scaled")

        if self._use_rope:
            head_dim = d_model // cfg.model.num_heads
            self._rope_sin, self._rope_cos = build_rope_tables(
                head_dim=head_dim,
                max_seq_len=cfg.model.max_seq_len,
            )
        else:
            self._rope_sin = None
            self._rope_cos = None

        # ------------------------------------------------------------
        # Transformer blocks
        # ------------------------------------------------------------
        self.blocks = [
            TransformerBlock(config=cfg)
            for _ in range(num_layers)
        ]

        self.final_norm = build_normalization(cfg)

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
    ) -> jnp.ndarray:

        # ------------------------------------------------------------
        # 🔴 CRITICAL: enforce input contract
        # ------------------------------------------------------------
        assert input_ids.ndim == 2, f"[GPT] Expected (B, T), got {input_ids.shape}"

        B, T = input_ids.shape

        # ------------------------------------------------------------
        # Token embedding
        # ------------------------------------------------------------
        x = self.token_embedding(input_ids)  # (B, T, D)
        x = x.astype(self._compute_dtype)

        # ------------------------------------------------------------
        # Positional encoding (safe broadcasting)
        # ------------------------------------------------------------
        if self.positional is not None:
            positions = jnp.arange(T)[None, :]  # (1, T)
            pos_emb = self.positional(positions)  # (1, T, D)

            # 🔴 CRITICAL FIX: enforce shape explicitly
            assert pos_emb.ndim == 3, f"[GPT] pos_emb wrong shape: {pos_emb.shape}"
            assert pos_emb.shape[1] == T, f"[GPT] pos_emb T mismatch: {pos_emb.shape}"

            # Safe broadcast
            x = x + pos_emb.astype(self._compute_dtype)

        # ------------------------------------------------------------
        # RoPE tables (slice once)
        # ------------------------------------------------------------
        rope_tables: Optional[Tuple] = None
        if self._use_rope:
            rope_tables = (
                self._rope_sin[:T],
                self._rope_cos[:T],
            )

        # ------------------------------------------------------------
        # Transformer stack
        # ------------------------------------------------------------
        for block in self.blocks:
            x = block(x, rope_tables=rope_tables, doc_ids=doc_ids)

        # ------------------------------------------------------------
        # Final norm
        # ------------------------------------------------------------
        x = self.final_norm(x)

        # ------------------------------------------------------------
        # Back to FP32 for logits
        # ------------------------------------------------------------
        x = x.astype(jnp.float32)

        # ------------------------------------------------------------
        # Output projection
        # ------------------------------------------------------------
        if self.config.architecture.weight_tying:
            embedding_table = self.token_embedding.embedding  # (V, D)
            logits = jnp.einsum("btd,vd->btv", x, embedding_table)
        else:
            logits = self.lm_head(x)

        return logits
