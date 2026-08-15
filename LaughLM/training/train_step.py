"""
LaughLM/training/train_step.py

PMAP-native train/eval step for LaughLM.

Design:
- replicated params/state
- local forward/backward per device
- direct fast path when grad_accum == 1
- scan accumulation path when grad_accum > 1
- cross-device grad averaging via pmean
- FP32 grad accumulation
- Optax optimizer updates

PMAP chunked-loss fix:
- Training/eval now request final hidden states from LlamaForCausalLM.
- Loss applies LM head inside loss.py.
- With loss.chunked_logits=True, full [B, T, vocab] logits are never
  materialized in the hot path.
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
)


Params = Any
Batch = jnp.ndarray
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
        "loss_backend": str(
            _loss_attr(
                loss_config,
                "backend",
                "native",
            )
        ),
        "tokamax_implementation": str(
            _loss_attr(
                loss_config,
                "tokamax_implementation",
                "mosaic_tpu",
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
# Train step
# ============================================================

def create_train_step(
    *,
    model,
    optimizer,
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

    # ========================================================
    # Loss fn
    # ========================================================

    def loss_fn(
        params,
        micro_batch,
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

        lm_head_kernel = _get_lm_head_kernel(
            params,
            tie_word_embeddings=tie_word_embeddings,
        )

        lm_head_bias = _get_lm_head_bias(
            params,
            tie_word_embeddings=tie_word_embeddings,
        )

        loss, metrics = compute_lm_loss_from_hidden(
            hidden_states=hidden_states,
            targets=targets,
            lm_head_kernel=lm_head_kernel,
            lm_head_bias=lm_head_bias,
            **loss_options,
        )

        return loss, metrics

    # ========================================================
    # Shared optimizer update
    # ========================================================

    def apply_update(
        state,
        params,
        grads,
        loss,
    ):
        grads = jax.lax.pmean(
            grads,
            axis_name="data",
        )

        loss = jax.lax.pmean(
            loss.astype(jnp.float32),
            axis_name="data",
        )

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

        updates, new_opt_state = (
            optimizer.update(
                grads,
                state.opt_state,
                params,
            )
        )

        new_params = optax.apply_updates(
            params,
            updates,
        )

        return (
            new_params,
            new_opt_state,
            loss,
            grad_norm,
        )

    # ========================================================
    # PMAP train step
    # ========================================================

    def train_step(
        state,
        batch,
    ):
        batch = constrain_batch(
            batch
        )

        params = state.params

        # ====================================================
        # Fast path
        # ====================================================

        if grad_accum == 1:
            micro_batch = batch[0]

            (
                (loss, _aux),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            grads = jax.tree_util.tree_map(
                lambda g:
                g.astype(jnp.float32),
                grads,
            )

        # ====================================================
        # Accumulation path
        # ====================================================

        else:
            grads_accum = (
                jax.tree_util.tree_map(
                    lambda p:
                    jnp.zeros_like(
                        p,
                        dtype=jnp.float32,
                    ),
                    params,
                )
            )

            def scan_fn(
                grads_accum,
                micro_batch,
            ):
                (
                    (loss, _aux),
                    grads,
                ) = jax.value_and_grad(
                    loss_fn,
                    has_aux=True,
                )(
                    params,
                    micro_batch,
                )

                grads_accum = (
                    jax.tree_util.tree_map(
                        lambda acc, g:
                        acc + g.astype(jnp.float32),
                        grads_accum,
                        grads,
                    )
                )

                return grads_accum, loss

            grads_accum, losses = jax.lax.scan(
                scan_fn,
                grads_accum,
                batch,
            )

            grads = jax.tree_util.tree_map(
                lambda g:
                g / jnp.asarray(
                    grad_accum,
                    dtype=jnp.float32,
                ),
                grads_accum,
            )

            loss = jnp.mean(
                losses,
                dtype=jnp.float32,
            )

        (
            new_params,
            new_opt_state,
            loss,
            grad_norm,
        ) = apply_update(
            state,
            params,
            grads,
            loss,
        )

        new_state = state.apply_grad_step(
            params=new_params,
            opt_state=new_opt_state,
        )

        metrics = {
            "loss": loss.astype(
                jnp.float32
            ),
            "grad_norm": grad_norm.astype(
                jnp.float32
            ),
        }

        return (
            new_state,
            metrics,
        )

    return jax.pmap(
        train_step,
        axis_name="data",
        donate_argnums=(0,),
    )


# ============================================================
# Eval step
# ============================================================

def create_eval_step(
    *,
    model,
    loss_config=None,
):
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

    def eval_step(
        state,
        batch,
    ):
        batch = constrain_batch(
            batch
        )

        inputs, targets = shift_tokens(
            batch
        )

        hidden_states, _ = model.apply(
            {"params": state.params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
            return_hidden=True,
        )

        lm_head_kernel = _get_lm_head_kernel(
            state.params,
            tie_word_embeddings=tie_word_embeddings,
        )

        lm_head_bias = _get_lm_head_bias(
            state.params,
            tie_word_embeddings=tie_word_embeddings,
        )

        loss, metrics = compute_lm_loss_from_hidden(
            hidden_states=hidden_states,
            targets=targets,
            lm_head_kernel=lm_head_kernel,
            lm_head_bias=lm_head_bias,
            **loss_options,
        )

        loss = jax.lax.pmean(
            loss.astype(jnp.float32),
            axis_name="data",
        )

        z_loss = jax.lax.pmean(
            metrics["z_loss"].astype(jnp.float32),
            axis_name="data",
        )

        return {
            "loss": loss,
            "z_loss": z_loss,
        }

    return jax.pmap(
        eval_step,
        axis_name="data",
    )
