"""
LaughLM/model/llama/model.py

Canonical Llama decoder-only language model.

Design goals
------------
- HF-compatible architecture semantics
- deterministic KV-cache behavior
- explicit train/prefill/decode modes
- minimal abstraction surface
- future-ready sharding compatibility

Tensor conventions
------------------
input_ids:
    [B, T]

hidden_states:
    [B, T, D]

attention_mask:
    [B, 1, Tq, Tk]

logits:
    [B, T, V]
"""

from typing import Optional

from flax import linen as nn

import jax.numpy as jnp

from LaughLM.model.llama.config import LlamaConfig
from LaughLM.model.llama.decoder import (
    LlamaDecoderLayer,
)
from LaughLM.model.llama.rmsnorm import RMSNorm
from LaughLM.model.llama.kv_cache import (
    KVCache,
)
from LaughLM.model.llama.masks import (
    build_causal_mask,
    build_decode_mask,
)


class LlamaModel(nn.Module):

    config: LlamaConfig

    def setup(self):

        config = self.config

        self.embed_tokens = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            name="embed_tokens",
        )

        self.layers = [
            LlamaDecoderLayer(
                config=config,
                name=f"layers_{i}",
            )
            for i in range(
                config.num_hidden_layers
            )
        ]

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            name="norm",
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        kv_caches: Optional[list[KVCache]] = None,
        mode: str = "train",
    ) -> tuple[
        jnp.ndarray,
        Optional[list[KVCache]],
    ]:
        """
        Parameters
        ----------
        input_ids:
            [B, T]

        positions:
            [B, T]

        mode:
            "train"
            "prefill"
            "decode"
        """

        config = self.config

        B, T = input_ids.shape

        hidden_states = self.embed_tokens(
            input_ids
        )

        # ──────────────────────────────────────────
        # Attention mask
        # ──────────────────────────────────────────

        if mode in ("train", "prefill"):

            attention_mask = build_causal_mask(
                query_length=T,
                key_length=T,
                dtype=hidden_states.dtype,
            )

        elif mode == "decode":

            if kv_caches is None:
                raise ValueError(
                    "decode mode requires kv_caches"
                )

            key_length = (
                kv_caches[0]
                .cache_position
            )

            attention_mask = build_decode_mask(
                query_length=T,
                key_length=key_length,
                dtype=hidden_states.dtype,
            )

        else:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # ──────────────────────────────────────────
        # Decoder stack
        # ──────────────────────────────────────────

        updated_caches = []

        for layer_idx, layer in enumerate(
            self.layers
        ):

            layer_cache = None

            if kv_caches is not None:
                layer_cache = kv_caches[
                    layer_idx
                ]

            hidden_states, updated_cache = (
                layer(
                    hidden_states=hidden_states,
                    positions=positions,
                    attention_mask=attention_mask,
                    kv_cache=layer_cache,
                    mode=mode,
                )
            )

            if kv_caches is not None:
                updated_caches.append(
                    updated_cache
                )

        hidden_states = self.norm(
            hidden_states
        )

        if kv_caches is None:
            updated_caches = None

        return (
            hidden_states,
            updated_caches,
        )


class LlamaForCausalLM(nn.Module):

    config: LlamaConfig

    def setup(self):

        config = self.config

        self.model = LlamaModel(
            config=config,
            name="model",
        )

        self.lm_head = nn.Dense(
            config.vocab_size,
            use_bias=False,
            name="lm_head",
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        positions: jnp.ndarray,
        kv_caches: Optional[list[KVCache]] = None,
        mode: str = "train",
    ) -> tuple[
        jnp.ndarray,
        Optional[list[KVCache]],
    ]:
        """
        Parameters
        ----------
        input_ids:
            [B, T]

        positions:
            [B, T]
        """

        hidden_states, updated_caches = (
            self.model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                mode=mode,
            )
        )

        # ──────────────────────────────────────────
        # LM head
        # ──────────────────────────────────────────

        if self.config.tie_word_embeddings:

            embedding = (
                self.model
                .embed_tokens
                .embedding
            )

            logits = jnp.einsum(
                "btd,vd->btv",
                hidden_states,
                embedding,
            )

        else:

            logits = self.lm_head(
                hidden_states
            )

        return logits, updated_caches