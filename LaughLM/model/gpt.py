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
    """
    Decoder-only language model (GPT / Llama architecture).

    Key design decisions
    --------------------
    1. RoPE is pre-computed once in setup() and passed to every
       TransformerBlock → attention layer, where it is applied to Q and K.
       It is NOT added to the input embeddings.

    2. Weight tying: the input token embedding matrix is reused as the
       output projection (lm_head). This is done by storing the embedding
       table in a variable and passing it explicitly to the logit computation,
       ensuring JAX autodiff correctly identifies it as a shared parameter
       and accumulates gradients from both uses.

    3. Mixed precision: the forward pass runs in bfloat16 on TPU. Parameters
       are stored in float32. Logits are cast back to float32 before the loss
       to avoid numerical issues in cross-entropy.
    """

    config: LaughLMConfig

    def setup(self):

        cfg         = self.config
        d_model     = cfg.model.d_model
        vocab_size  = cfg.model.vocab_size
        num_layers  = cfg.model.num_layers
        pos_type    = cfg.architecture.positional
        compute_bf16 = (cfg.parallelism.compute_dtype == "bfloat16")

        self._compute_dtype = jnp.bfloat16 if compute_bf16 else jnp.float32

        # ------------------------------------------------------------------
        # Token embedding
        # ------------------------------------------------------------------
        self.token_embedding = nn.Embed(
            num_embeddings=vocab_size,
            features=d_model,
            embedding_init=nn.initializers.normal(
                stddev=cfg.initialization.embedding_std
            ),
        )

        # ------------------------------------------------------------------
        # Additive positional encoding (learned / sinusoidal)
        # Returns None for RoPE — handled separately below.
        # ------------------------------------------------------------------
        self.positional = build_positional_encoding(cfg)

        # ------------------------------------------------------------------
        # RoPE tables (pre-computed, not learned parameters)
        # Only built when positional = rope or rope_scaled.
        # ------------------------------------------------------------------
        self._use_rope = pos_type in ("rope", "rope_scaled")

        if self._use_rope:
            head_dim = d_model // cfg.model.num_heads
            # build_rope_tables returns (sin, cos) both [max_seq_len, head_dim // 2]
            # These are JAX arrays, not nn.Module params — they don't appear
            # in the parameter dict and are not updated by the optimizer.
            self._rope_sin, self._rope_cos = build_rope_tables(
                head_dim=head_dim,
                max_seq_len=cfg.model.max_seq_len,
            )
        else:
            self._rope_sin = None
            self._rope_cos = None

        # ------------------------------------------------------------------
        # Transformer stack
        # ------------------------------------------------------------------
        self.blocks = [
            TransformerBlock(config=cfg)
            for _ in range(num_layers)
        ]

        # ------------------------------------------------------------------
        # Final layer norm
        # ------------------------------------------------------------------
        self.final_norm = build_normalization(cfg)

        # ------------------------------------------------------------------
        # Output projection (only when NOT using weight tying)
        # ------------------------------------------------------------------
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
        """
        Forward pass.

        Parameters
        ----------
        input_ids : [B, T] integer token IDs
        doc_ids   : [B, T] integer document IDs for cross-doc masking.
                    Required when config.data.packing=True.
                    When None, the standard triangular causal mask is used.

        Returns
        -------
        logits : [B, T, vocab_size] float32
        """

        T = input_ids.shape[1]

        # ------------------------------------------------------------------
        # Embed tokens
        # ------------------------------------------------------------------
        x = self.token_embedding(input_ids)        # [B, T, D]  float32

        # Cast to compute dtype (bfloat16 on TPU for speed)
        x = x.astype(self._compute_dtype)

        # ------------------------------------------------------------------
        # Additive positional encoding (learned / sinusoidal only)
        # RoPE is threaded to attention layers — not added here.
        # ------------------------------------------------------------------
        if self.positional is not None:
            positions = jnp.arange(T)[None, :]     # [1, T]
            pos_emb = self.positional(positions)   # [1, T, D]
            x = x + pos_emb.astype(self._compute_dtype)

        # ------------------------------------------------------------------
        # Build RoPE tables for this sequence length
        # Slice the pre-computed tables to current T (saves compute).
        # ------------------------------------------------------------------
        rope_tables: Optional[Tuple] = None
        if self._use_rope:
            rope_tables = (
                self._rope_sin[:T],   # [T, head_dim // 2]
                self._rope_cos[:T],
            )

        # ------------------------------------------------------------------
        # Transformer layers
        # ------------------------------------------------------------------
        for block in self.blocks:
            x = block(x, rope_tables=rope_tables, doc_ids=doc_ids)

        # ------------------------------------------------------------------
        # Final norm
        # ------------------------------------------------------------------
        x = self.final_norm(x)

        # Cast back to float32 before logit projection
        # (softmax / cross-entropy need float32 precision)
        x = x.astype(jnp.float32)

        # ------------------------------------------------------------------
        # Logit projection
        # ------------------------------------------------------------------
        if self.config.architecture.weight_tying:
            # Weight tying: reuse the token embedding matrix as the output
            # projection. Transposed: [D] → [V].
            #
            # We access the embedding via self.token_embedding.embedding
            # (Flax nn.Embed stores the table at this attribute).
            # This is the correct Flax idiom — JAX autodiff will correctly
            # accumulate gradients through this reference because both uses
            # (embedding lookup and logit projection) refer to the same
            # parameter array in the variable collection.
            embedding_table = self.token_embedding.embedding   # [V, D]
            logits = jnp.einsum("btd,vd->btv", x, embedding_table)

        else:
            logits = self.lm_head(x)

        return logits   # [B, T, V]  float32
