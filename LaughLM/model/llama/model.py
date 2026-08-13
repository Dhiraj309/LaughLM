"""
Canonical Llama decoder-only language model.

PMAP chunked-loss fix:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
LlamaForCausalLM can return final hidden states before the LM head
via return_hidden=True. This lets training compute exact chunked CE
without materializing full [B, T, vocab] logits.

Default behavior remains unchanged:
return_hidden=False returns full logits for HF export, generation,
and logits parity validation.
"""

from typing import Optional

import jax.numpy as jnp
from flax import linen as nn

from LaughLM.model.llama.config import LlamaConfig

from LaughLM.model.llama.initialization import (
    create_dense,
    create_embedding,
    constrain_hidden_states,
)

from LaughLM.distributed.sharding import constrain_logits

from LaughLM.model.llama.decoder import LlamaDecoderLayer
from LaughLM.model.llama.remat import remat_module
from LaughLM.model.llama.rmsnorm import RMSNorm
from LaughLM.model.llama.kv_cache import KVCache

from LaughLM.model.llama.masks import (
    build_causal_mask,
    build_decode_mask,
)


# ============================================================
# Helpers
# ============================================================

def _uses_splash_attention(
    config: LlamaConfig,
) -> bool:
    return (
        getattr(
            config,
            "attention_impl",
            "standard",
        )
        == "splash"
    )


# ============================================================
# Scanned decoder block
# ============================================================

from LaughLM.utils.sharding_factory import apply_rematerialization, get_remat_policy

class ScannedDecoderLayer(nn.Module):
    config: LlamaConfig

    def setup(self):
        self.layer = LlamaDecoderLayer(
            config=self.config,
            name="block",
        )

    def __call__(
        self,
        hidden_states,
        positions,
        attention_mask,
        kv_cache,
        mode,
    ):
        return self.layer(
            hidden_states=hidden_states,
            positions=positions,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            mode=mode,
        )


# ============================================================
# Base model
# ============================================================

class LlamaModel(nn.Module):
    config: LlamaConfig

    def setup(self):
        config = self.config

        self.embed_tokens = create_embedding(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            config=config,
            name="embed_tokens",
        )

        LayerCls = ScannedDecoderLayer

        if getattr(config, "remat_policy", None) is not None:
            LayerCls = remat_module(
                ScannedDecoderLayer,
                policy=config.remat_policy,
                prevent_cse=getattr(
                    config,
                    "prevent_cse",
                    False,
                ),
            )

        if getattr(config, "scan_layers", False):
            ScanLayer = nn.scan(
                LayerCls,
                variable_axes={
                    "params": 0,
                },
                variable_broadcast={
                    "cache",
                },
                split_rngs={
                    "params": True,
                    "dropout": True,
                },
                # `hidden_states` is the scan carry. `in_axes` therefore
                # describes only the remaining four scan-body inputs. All of
                # them are shared by every decoder layer during training;
                # scanning them would incorrectly require a leading layer axis
                # and produces a carry/xs pytree-length mismatch.
                in_axes=(
                    nn.broadcast,  # positions
                    nn.broadcast,  # attention_mask
                    nn.broadcast,  # kv_cache
                    nn.broadcast,  # mode
                ),
                out_axes=0,
                length=config.num_hidden_layers,
                metadata_params={
                    "partition_name": "layers",
                },
            )

            self.layers = ScanLayer(
                config=config,
                name="layers",
            )

        else:
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
            dtype=config.compute_dtype,
            param_dtype=config.param_dtype,
            name="norm",
        )

    def __call__(
        self,
        input_ids: jnp.ndarray,
        position_ids: Optional[jnp.ndarray] = None,
        kv_caches: Optional[list[KVCache]] = None,
        use_cache: bool = False,
        mode: str = "train",
    ) -> tuple[jnp.ndarray, Optional[list[KVCache]]]:

        if self.config.scan_layers and use_cache:
            raise ValueError(
                "scan_layers with KV cache is not yet supported"
            )

        B, T = input_ids.shape

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
            if mode == "decode" and kv_caches is not None:
                start_pos = kv_caches[0].cache_position
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

        if mode in ("train", "prefill"):
            if _uses_splash_attention(
                self.config
            ):
                attention_mask = None
            else:
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
                kv_caches[0].cache_position
                + T
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

        # --------------------------------------------------
        # Decoder stack
        # --------------------------------------------------

        if getattr(
            self.config,
            "scan_layers",
            False,
        ):
            hidden_states, _ = self.layers(
                hidden_states,
                position_ids,
                attention_mask,
                None,
                mode,
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

                hidden_states, updated_cache = layer(
                    hidden_states=hidden_states,
                    positions=position_ids,
                    attention_mask=attention_mask,
                    kv_cache=layer_cache,
                    mode=mode,
                )

                hidden_states = constrain_hidden_states(
                    hidden_states
                )

                if use_cache:
                    updated_caches.append(
                        updated_cache
                    )

        hidden_states = self.norm(
            hidden_states
        )

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        if not use_cache:
            updated_caches = None

        return hidden_states, updated_caches


# ============================================================
# Causal LM wrapper
# ============================================================

class LlamaForCausalLM(nn.Module):
    config: LlamaConfig

    def setup(self):
        config = self.config

        self.model = LlamaModel(
            config=config,
            name="model",
        )

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
        return_hidden: bool = False,
    ) -> tuple[jnp.ndarray, Optional[list[KVCache]]]:

        hidden_states, updated_caches = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            kv_caches=kv_caches,
            use_cache=use_cache,
            mode=mode,
        )

        # --------------------------------------------------
        # Training loss fast path
        #
        # Return final normalized hidden states before the LM head.
        # This avoids full [B, T, vocab] materialization.
        # --------------------------------------------------

        if return_hidden:
            return hidden_states, updated_caches

        # --------------------------------------------------
        # Standard full-logits path
        # --------------------------------------------------

        if self.config.tie_word_embeddings:
            embedding = self.model.embed_tokens.embedding

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

        logits = logits.astype(
            jnp.float32
        )

        logits = constrain_logits(
            logits
        )

        logits = logits.astype(
            self.config.output_dtype
        )

        return logits, updated_caches
