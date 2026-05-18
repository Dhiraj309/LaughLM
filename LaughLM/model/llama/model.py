# LaughLM/model/llama/model.py

"""
Canonical Llama decoder-only language model.

Frontier-grade SPMD additions:
────────────────────────────────────────────────────
1. Logical partitioning for embeddings/logits
2. Hidden-state logical constraints
3. Tensor-parallel-safe tied embeddings
4. Remat-ready decoder stack
5. Scan-compatible layer structure
6. Vocab-axis sharding semantics
7. BF16 compute + FP32 logits stability
8. Future-ready GSPMD compatibility
9. nn.scan transformer stack (MaxText/T5X-style)

Design goals
------------
- HF-compatible architecture semantics
- deterministic KV-cache behavior
- explicit train/prefill/decode modes
- deterministic initialization semantics
- rematerialization-ready transformer stack
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
    constrain_hidden_states,
)

from LaughLM.distributed.sharding import (
    constrain_logits,
)

from LaughLM.model.llama.decoder import (
    LlamaDecoderLayer,
)

from LaughLM.model.llama.remat import (
    remat_module,
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
        # Optional rematerialization
        # --------------------------------------------------

        LayerCls = LlamaDecoderLayer

        if getattr(
            config,
            "remat_policy",
            None,
        ) is not None:

            LayerCls = remat_module(
                LlamaDecoderLayer,
                policy=config.remat_policy,
                prevent_cse=getattr(
                    config,
                    "prevent_cse",
                    False,
                ),
            )

        # --------------------------------------------------
        # Decoder stack
        # --------------------------------------------------
        #
        # IMPORTANT
        # ─────────────────────────────────────────────
        # Current scan implementation supports:
        #
        # - training
        # - prefill
        #
        # It does NOT yet support:
        #
        # - autoregressive KV-cache decode
        #
        # because scanned mutable cache carries
        # require a more advanced carry structure.
        #

        if getattr(
            config,
            "scan_layers",
            False,
        ):

            ScanLayer = nn.scan(
                LayerCls,

                variable_axes={
                    "params": 0,
                },

                split_rngs={
                    "params": True,
                },

                in_axes=nn.broadcast,
                out_axes=nn.broadcast,

                length=config.num_hidden_layers,
            )

            self.layers = ScanLayer(
                config=config,
                name="layers",
            )

        else:

            self.layers = [
                LayerCls(
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

        # --------------------------------------------------
        # Scan limitation
        # --------------------------------------------------

        if (
            getattr(
                self.config,
                "scan_layers",
                False,
            )
            and use_cache
        ):
            raise ValueError(
                "scan_layers with KV cache "
                "is not yet supported"
            )

        B, T = input_ids.shape

        # --------------------------------------------------
        # Token embeddings
        # --------------------------------------------------

        hidden_states = self.embed_tokens(
            input_ids
        )

        hidden_states = constrain_hidden_states(
            hidden_states
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

            position_ids = jnp.broadcast_to(
                jnp.arange(
                    start_pos,
                    start_pos + T,
                    dtype=jnp.int32,
                )[None, :],
                (B, T),
            )

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

        if getattr(
            self.config,
            "scan_layers",
            False,
        ):

            hidden_states, _ = (
                self.layers(
                    hidden_states=hidden_states,
                    positions=position_ids,
                    attention_mask=attention_mask,
                    kv_cache=None,
                    mode=mode,
                )
            )

            hidden_states = constrain_hidden_states(
                hidden_states
            )

            updated_caches = None

        else:

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

                hidden_states = (
                    constrain_hidden_states(
                        hidden_states
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

        hidden_states = constrain_hidden_states(
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
                preferred_element_type=jnp.float32,
            )

        else:

            logits = self.lm_head(
                hidden_states
            )

        # --------------------------------------------------
        # Stable fp32 logits
        # --------------------------------------------------

        logits = logits.astype(
            jnp.float32
        )

        # --------------------------------------------------
        # Logical logits constraint
        # --------------------------------------------------

        logits = constrain_logits(
            logits
        )

        # --------------------------------------------------
        # Stable output dtype
        # --------------------------------------------------

        logits = logits.astype(
            self.config.output_dtype
        )

        return (
            logits,
            updated_caches,
        )
