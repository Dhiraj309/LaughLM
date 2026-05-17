"""
LaughLM/model/llama/model.py

Canonical Llama decoder-only language model.

Design goals
------------
- HF-compatible architecture semantics
- deterministic KV-cache behavior
- explicit train/prefill/decode modes
- deterministic initialization semantics
- minimal abstraction surface
- future-ready sharding compatibility

Tensor conventions
------------------
input_ids:
    [B, T]

position_ids:
    [B, T]

hidden_states:
    [B, T, D]

attention_mask:
    [B, 1, Tq, Tk]

logits:
    [B, T, V]

KV cache:
    per-layer static cache:
        key/value:
            [B, S, KVH, Dh]
"""

from typing import Optional

import jax.numpy as jnp

from flax import linen as nn

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.initialization import (
    create_dense,
    create_embedding,
)

from LaughLM.model.llama.decoder import (
    LlamaDecoderLayer,
)

from LaughLM.model.llama.rmsnorm import (
    RMSNorm,
)

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

        # --------------------------------------------------
        # Token embeddings
        # --------------------------------------------------

        self.embed_tokens = create_embedding(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            config=config,
            name="embed_tokens",
        )

        # --------------------------------------------------
        # Decoder layers
        # --------------------------------------------------
        #
        # IMPORTANT
        # ---------
        # Explicit stable layer names:
        #
        # model.layers_0
        # model.layers_1
        #
        # Required for:
        # - deterministic checkpoints
        # - stable traversal
        # - remat/scan compatibility
        # - HF conversion
        #

        self.layers = [
            LlamaDecoderLayer(
                config=config,
                name=f"layers_{i}",
            )
            for i in range(
                config.num_hidden_layers
            )
        ]

        # --------------------------------------------------
        # Final RMSNorm
        # --------------------------------------------------

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            name="norm",
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[list[KVCache]] = None,
        use_cache: bool = False,
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

        position_ids:
            [B, T]

        kv_caches:
            list[KVCache]

        use_cache:
            Whether to return updated caches.

        mode:
            "train"
            "prefill"
            "decode"
        """

        B, T = input_ids.shape

        hidden_states = self.embed_tokens(
            input_ids
        )

        # --------------------------------------------------
        # Position IDs
        # --------------------------------------------------

        if position_ids is None:

            if (
                mode == "decode"
                and kv_caches is not None
            ):

                start_pos = (
                    kv_caches[0]
                    .cache_position
                )

            else:

                start_pos = 0

            position_ids = jnp.arange(
                start_pos,
                start_pos + T,
                dtype=jnp.int32,
            )[None, :]

        # --------------------------------------------------
        # Attention mask
        # --------------------------------------------------

        if mode in (
            "train",
            "prefill",
        ):

            attention_mask = (
                build_causal_mask(
                    query_length=T,
                    key_length=T,
                    dtype=hidden_states.dtype,
                )
            )

        elif mode == "decode":

            if kv_caches is None:

                raise ValueError(
                    "decode mode requires kv_caches"
                )

            #
            # IMPORTANT
            # ---------
            # During decode:
            #
            # visible KV length =
            # existing cache + current token(s)
            #

            key_length = (
                kv_caches[0]
                .cache_position
                + T
            )

            attention_mask = (
                build_decode_mask(
                    query_length=T,
                    key_length=key_length,
                    dtype=hidden_states.dtype,
                )
            )

        else:

            raise ValueError(
                f"Unknown mode: {mode}"
            )

        # --------------------------------------------------
        # Decoder stack
        # --------------------------------------------------

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
                    positions=position_ids,
                    attention_mask=attention_mask,
                    kv_cache=layer_cache,
                    mode=mode,
                )
            )

            if use_cache:

                updated_caches.append(
                    updated_cache
                )

        # --------------------------------------------------
        # Final normalization
        # --------------------------------------------------

        hidden_states = self.norm(
            hidden_states
        )

        if not use_cache:

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

        # --------------------------------------------------
        # LM head
        # --------------------------------------------------
        #
        # HF Llama uses the same Gaussian init:
        #
        # std = config.initializer_range
        #

        self.lm_head = create_dense(
            features=config.vocab_size,
            config=config,
            use_bias=False,
            name="lm_head",
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[list[KVCache]] = None,
        use_cache: bool = False,
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

        position_ids:
            [B, T]

        Returns
        -------
        logits:
            [B, T, vocab_size]
        """

        hidden_states, updated_caches = (
            self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                kv_caches=kv_caches,
                use_cache=use_cache,
                mode=mode,
            )
        )

        # --------------------------------------------------
        # LM head
        # --------------------------------------------------

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

        return (
            logits,
            updated_caches,
        )
