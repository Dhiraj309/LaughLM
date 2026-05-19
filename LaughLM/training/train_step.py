"""
LaughLM/training/train_step.py

Frontier-grade mesh-native train step.

2026 TPU/FSDP upgrades:
────────────────────────────────────────────
1. Correct cross-device gradient reduction
2. Global grad norm across mesh
3. TPU-safe accumulation semantics
4. Correct replicated metric reduction
5. Stable FSDP-compatible optimizer updates
6. Future tensor-parallel compatibility
7. No silent multi-host desync
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import optax

from flax.linen import (
    partitioning as nn_partitioning,
)

from LaughLM.training.loss import (
    shift_tokens,
    compute_loss,
)

from LaughLM.distributed.sharding import (
    get_logical_axis_rules,
    constrain_batch,
    constrain_logits,
    replicated_sharding,
)


Params = Any
Batch = jnp.ndarray
Metrics = Dict[str, jnp.ndarray]


# ============================================================
# Train step
# ============================================================

def create_train_step(
    *,
    model,
    optimizer,
    config,
    mesh,
    state_shardings,
    data_sharding,
    grad_accum: int,
    max_grad_norm: float = 1.0,
):

    metrics_sharding = (
        replicated_sharding(mesh)
    )

    # --------------------------------------------------------
    # Mesh axes
    # --------------------------------------------------------

    mesh_axes = tuple(mesh.axis_names)

    # --------------------------------------------------------
    # Loss fn
    # --------------------------------------------------------

    def loss_fn(
        params,
        micro_batch,
    ):

        inputs, targets = shift_tokens(
            micro_batch
        )

        logits, _ = model.apply(
            {"params": params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, metrics = compute_loss(
            logits,
            targets,
        )

        return loss, metrics

    # --------------------------------------------------------
    # Train step
    # --------------------------------------------------------

    def train_step(
        state,
        batch,
    ):

        batch = constrain_batch(
            batch
        )

        params = state.params

        # ----------------------------------------------------
        # Stateless step RNG
        # ----------------------------------------------------

        step_rng = jax.random.fold_in(
            state.rng_key,
            state.step,
        )

        # ----------------------------------------------------
        # FP32 grad accumulation
        # ----------------------------------------------------

        grads_accum = (
            jax.tree_util.tree_map(
                lambda p: jnp.zeros_like(
                    p,
                    dtype=jnp.float32,
                ),
                params,
            )
        )

        # ----------------------------------------------------
        # Microbatch scan
        # ----------------------------------------------------

        def scan_fn(
            carry,
            micro_batch,
        ):

            grads_accum, rng = carry

            rng, _ = jax.random.split(
                rng
            )

            (
                (loss, _metrics),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            # ------------------------------------------------
            # Accumulate fp32 grads
            # ------------------------------------------------

            grads_accum = (
                jax.tree_util.tree_map(
                    lambda acc, g:
                    acc + g.astype(jnp.float32),
                    grads_accum,
                    grads,
                )
            )

            return (
                (grads_accum, rng),
                loss,
            )

        (
            (grads_accum, _),
            losses,
        ) = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
            batch,
        )

        # ----------------------------------------------------
        # Mean microbatch grads
        # ----------------------------------------------------

        grads = jax.tree_util.tree_map(
            lambda g:
            g / jnp.asarray(
                grad_accum,
                dtype=jnp.float32,
            ),
            grads_accum,
        )

        # ====================================================
        # GLOBAL gradient reduction
        # ====================================================

        if len(mesh_axes) > 0:

            grads = jax.lax.pmean(
                grads,
                axis_name=mesh_axes,
            )

        # ----------------------------------------------------
        # Mean loss
        # ----------------------------------------------------

        loss = jnp.mean(
            losses,
            dtype=jnp.float32,
        )

        # ====================================================
        # GLOBAL loss reduction
        # ====================================================

        if len(mesh_axes) > 0:

            loss = jax.lax.pmean(
                loss,
                axis_name=mesh_axes,
            )

        # ====================================================
        # Global grad norm
        # ====================================================

        grad_norm = optax.global_norm(
            grads
        )

        if len(mesh_axes) > 0:

            grad_norm = jax.lax.pmean(
                grad_norm,
                axis_name=mesh_axes,
            )

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm
            / jnp.maximum(
                grad_norm,
                1e-6,
            ),
        )

        grads = jax.tree_util.tree_map(
            lambda g:
            g * clip_scale,
            grads,
        )

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        updates, new_opt_state = (
            optimizer.update(
                grads,
                state.opt_state,
                params,
            )
        )

        new_params = (
            optax.apply_updates(
                params,
                updates,
            )
        )

        # ----------------------------------------------------
        # Tokens processed
        # ----------------------------------------------------

        tokens_in_step = (
            jnp.asarray(
                batch.shape[0]
                * batch.shape[1]
                * batch.shape[2],
                dtype=jnp.int32,
            )
        )

        # ====================================================
        # GLOBAL token accounting
        # ====================================================

        if len(mesh_axes) > 0:

            tokens_in_step = (
                tokens_in_step
                * jax.device_count()
            )

        # ----------------------------------------------------
        # State update
        # ----------------------------------------------------

        new_state = (
            state.apply_grad_step(
                params=new_params,
                opt_state=new_opt_state,
                tokens_in_step=tokens_in_step,
            )
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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

    # ========================================================
    # Mesh-native compilation
    # ========================================================

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        return jax.jit(
            train_step,

            in_shardings=(
                state_shardings,
                data_sharding,
            ),

            out_shardings=(
                state_shardings,
                metrics_sharding,
            ),

            donate_argnums=(0,),
        )


# ============================================================
# Eval step
# ============================================================

def create_eval_step(
    *,
    model,
    config,
    mesh,
    state_shardings,
    data_sharding,
):

    metrics_sharding = (
        replicated_sharding(mesh)
    )

    mesh_axes = tuple(mesh.axis_names)

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

        logits, _ = model.apply(
            {"params": state.params},
            input_ids=inputs,
            use_cache=False,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, _ = compute_loss(
            logits,
            targets,
        )

        loss = loss.astype(
            jnp.float32
        )

        if len(mesh_axes) > 0:

            loss = jax.lax.pmean(
                loss,
                axis_name=mesh_axes,
            )

        return {
            "loss": loss,
        }

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        return jax.jit(
            eval_step,

            in_shardings=(
                state_shardings,
                data_sharding,
            ),

            out_shardings=metrics_sharding,
        )
