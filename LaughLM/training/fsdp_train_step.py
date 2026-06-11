"""
LaughLM/training/fsdp_train_step.py

GSPMD/FSDP train step.

No pmap.
No lax.pmean.
Global array semantics + shardings handle collectives.

Phase 4 parity fix:
- Match PMAP hidden-state LM loss path.
- Avoid materializing [B, T, vocab] when loss.chunked_logits=True.
- Respect loss config:
    chunked_logits
    logits_chunk_size
    remat_logits_chunks
    z_loss
    ignore_index
- Handle tied and untied LM heads explicitly.

Phase 4F optimization:
- Keep grad-accum scan carry minimal.
- Carry only accumulated gradients and scalar loss.
- Use loss-only hidden-state LM path in FSDP hot step.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from LaughLM.training.loss import (
    shift_tokens,
    compute_lm_loss_from_hidden,
)
from LaughLM.distributed.sharding import (
    constrain_batch,
    constrain_hidden_states,
)


Params = Any
Metrics = Dict[str, jnp.ndarray]


# ============================================================
# Loss config helpers
# ============================================================

def _loss_attr(
    loss_config,
    name: str,
    default,
):
    if loss_config is None:
        return default

    return getattr(
        loss_config,
        name,
        default,
    )


def _loss_kwargs(
    loss_config,
):
    return {
        "chunked_logits": bool(
            _loss_attr(
                loss_config,
                "chunked_logits",
                False,
            )
        ),
        "logits_chunk_size": int(
            _loss_attr(
                loss_config,
                "logits_chunk_size",
                4096,
            )
        ),
        "remat_logits_chunks": bool(
            _loss_attr(
                loss_config,
                "remat_logits_chunks",
                True,
            )
        ),
        "z_loss": float(
            _loss_attr(
                loss_config,
                "z_loss",
                1e-4,
            )
        ),
        "ignore_index": int(
            _loss_attr(
                loss_config,
                "ignore_index",
                -100,
            )
        ),
    }


def _unbox_param_leaf(x):
    """
    Flax logical partition wrappers may appear in params depending on
    initialization/restoration path. The training loss needs raw arrays.
    """

    if hasattr(x, "unbox"):
        try:
            return x.unbox(
                apply_constraint=False,
            )

        except TypeError:
            return x.unbox()

    if isinstance(x, dict) and "value" in x:
        return x["value"]

    return x


def _get_lm_head_kernel(
    params,
    *,
    tie_word_embeddings: bool,
):
    """
    Return the LM projection weight.

    Current production tied path:
      params["model"]["embed_tokens"]["embedding"]  # [vocab, hidden]

    Untied path:
      params["lm_head"]["kernel"]                   # [hidden, vocab]
    """

    if tie_word_embeddings:
        return _unbox_param_leaf(
            params["model"]["embed_tokens"]["embedding"]
        )

    return _unbox_param_leaf(
        params["lm_head"]["kernel"]
    )


def _get_lm_head_bias(
    params,
    *,
    tie_word_embeddings: bool,
):
    if tie_word_embeddings:
        return None

    lm_head = params.get(
        "lm_head",
        {},
    )

    bias = lm_head.get(
        "bias",
        None,
    )

    if bias is None:
        return None

    return _unbox_param_leaf(
        bias
    )


# ============================================================
# FSDP train step
# ============================================================

def create_fsdp_train_step(
    *,
    model,
    optimizer,
    state_sharding,
    batch_sharding,
    metrics_sharding,
    grad_accum: int,
    max_grad_norm: float = 1.0,
    loss_config=None,
):
    if grad_accum <= 0:
        raise ValueError(
            "grad_accum must be > 0"
        )

    loss_options = _loss_kwargs(
        loss_config
    )

    tie_word_embeddings = bool(
        getattr(
            model.config,
            "tie_word_embeddings",
            False,
        )
    )

    def loss_only_fn(
        params: Params,
        micro_batch: jnp.ndarray,
    ):
        inputs, targets = shift_tokens(
            micro_batch
        )

        hidden_states, _ = model.apply(
            {"params": params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
            return_hidden=True,
        )

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        lm_head_kernel = _get_lm_head_kernel(
            params,
            tie_word_embeddings=tie_word_embeddings,
        )

        lm_head_bias = _get_lm_head_bias(
            params,
            tie_word_embeddings=tie_word_embeddings,
        )

        loss, _ = compute_lm_loss_from_hidden(
            hidden_states=hidden_states,
            targets=targets,
            lm_head_kernel=lm_head_kernel,
            lm_head_bias=lm_head_bias,
            **loss_options,
        )

        return loss.astype(
            jnp.float32
        )

    def train_step(
        state,
        batch,
    ):
        batch = constrain_batch(
            batch
        )

        params = state.params

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(
                p,
                dtype=jnp.float32,
            ),
            params,
        )

        init_loss_sum = jnp.asarray(
            0.0,
            dtype=jnp.float32,
        )

        def scan_fn(
            carry,
            micro_batch,
        ):
            grads_accum, loss_sum = carry

            loss, grads = jax.value_and_grad(
                loss_only_fn
            )(
                params,
                micro_batch,
            )

            grads_accum = jax.tree_util.tree_map(
                lambda acc, g: acc + g.astype(jnp.float32),
                grads_accum,
                grads,
            )

            loss_sum = (
                loss_sum
                + loss.astype(jnp.float32)
            )

            return (
                grads_accum,
                loss_sum,
            ), None

        (
            grads_accum,
            loss_sum,
        ), _ = jax.lax.scan(
            scan_fn,
            (
                grads_accum,
                init_loss_sum,
            ),
            batch,
        )

        grad_accum_f32 = jnp.asarray(
            grad_accum,
            dtype=jnp.float32,
        )

        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum_f32,
            grads_accum,
        )

        loss = (
            loss_sum
            / grad_accum_f32
        ).astype(jnp.float32)

        grad_norm = optax.global_norm(
            grads
        ).astype(jnp.float32)

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm
            / jnp.maximum(
                grad_norm,
                1e-6,
            ),
        )

        grads = jax.tree_util.tree_map(
            lambda g: g * clip_scale,
            grads,
        )

        updates, new_opt_state = optimizer.update(
            grads,
            state.opt_state,
            params,
        )

        new_params = optax.apply_updates(
            params,
            updates,
        )

        tokens_in_step = jnp.asarray(
            batch.shape[0]
            * batch.shape[1]
            * batch.shape[2],
            dtype=jnp.int32,
        )

        new_state = state.apply_grad_step(
            params=new_params,
            opt_state=new_opt_state,
            tokens_in_step=tokens_in_step,
        )

        metrics: Metrics = {
            "loss": loss.astype(jnp.float32),
            "grad_norm": grad_norm.astype(jnp.float32),
        }

        return new_state, metrics

    return jax.jit(
        train_step,
        in_shardings=(state_sharding, batch_sharding),
        out_shardings=(state_sharding, metrics_sharding),
        donate_argnums=(0,),
    )
