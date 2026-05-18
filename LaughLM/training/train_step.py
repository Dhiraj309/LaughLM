"""
LaughLM/training/train_step.py

Frontier-grade GSPMD training step.

Architecture
────────────────────────────────────────────────────────
- Pure GSPMD / SPMD execution
- No pmap
- No pmean
- Global sharded arrays
- Mesh-aware jit compilation
- Explicit sharding contracts
- Float32 gradient accumulation
- Static-shape compilation
- XLA collective insertion via partitioner
- No hidden host synchronization
- Stateless RNG semantics

Inspired by:
- MaxText
- T5X
- Levanter
- Pax
"""

from __future__ import annotations

from typing import Any, Callable, Dict

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
    constrain_loss_tensor,
)


Params = Any
OptState = Any
Batch = jnp.ndarray

Metrics = Dict[str, jnp.ndarray]


# ============================================================
# Train Step
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
) -> Callable:
    """
    Create fully-sharded GSPMD training step.

    Returns
    -------
    jitted_train_step:
        (state, batch) -> (new_state, metrics)

    Batch shape:
        [grad_accum, global_batch, seq_len]

    IMPORTANT
    ─────────────────────────────────────────────
    This function operates on GLOBAL arrays.

    XLA/GSPMD automatically inserts collectives
    from logical shardings and PartitionSpecs.
    """

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    def loss_fn(
        params: Params,
        micro_batch: Batch,
    ):
        """
        Single microbatch loss.
        """

        # ----------------------------------------------------
        # Token shifting
        # ----------------------------------------------------

        inputs, targets = shift_tokens(
            micro_batch
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, _ = model.apply(
            {"params": params},
            inputs,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        # ----------------------------------------------------
        # Cross entropy
        # ----------------------------------------------------

        loss, metrics = compute_loss(
            logits,
            targets,
        )

        loss = constrain_loss_tensor(
            loss
        )

        return loss, metrics

    # --------------------------------------------------------
    # Train step
    # --------------------------------------------------------

    def train_step(
        state,
        batch,
    ):
        """
        Single optimizer step.

        Parameters
        ----------
        state:
            Global sharded TrainState

        batch:
            [grad_accum, global_batch, seq_len]
        """

        # ----------------------------------------------------
        # Batch logical constraints
        # ----------------------------------------------------

        batch = constrain_batch(
            batch
        )

        params = state.params

        opt_state = state.opt_state

        # ----------------------------------------------------
        # Stateless RNG
        #
        # Fold step into base key.
        #
        # Frontier standard:
        # avoids mutable RNG-in-state semantics.
        # ----------------------------------------------------

        step_rng = jax.random.fold_in(
            state.rng_key,
            state.step,
        )

        # ----------------------------------------------------
        # FP32 gradient accumulator
        #
        # Mandatory for stable bf16 training.
        # ----------------------------------------------------

        grads_accum = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(
                p,
                dtype=jnp.float32,
            ),
            params,
        )

        # ----------------------------------------------------
        # Gradient accumulation scan
        # ----------------------------------------------------

        def scan_fn(
            carry,
            micro_batch,
        ):

            grads_accum, rng = carry

            rng, subkey = jax.random.split(
                rng
            )

            # ------------------------------------------------
            # Forward + backward
            # ------------------------------------------------

            (
                (loss, metrics),
                grads,
            ) = jax.value_and_grad(
                loss_fn,
                has_aux=True,
            )(
                params,
                micro_batch,
            )

            # ------------------------------------------------
            # Accumulate grads in fp32
            # ------------------------------------------------

            grads_accum = (
                jax.tree_util.tree_map(
                    lambda g_acc, g:
                    g_acc + g.astype(jnp.float32),
                    grads_accum,
                    grads,
                )
            )

            return (
                (grads_accum, rng),
                (loss, metrics),
            )

        (
            (grads_accum, _),
            (losses, metrics_list),
        ) = jax.lax.scan(
            scan_fn,
            (grads_accum, step_rng),
            batch,
        )

        # ----------------------------------------------------
        # Average accumulated grads
        # ----------------------------------------------------

        grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum,
            grads_accum,
        )

        loss = jnp.mean(
            losses,
            dtype=jnp.float32,
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # No pmean here.
        #
        # GSPMD inserts collectives automatically
        # from PartitionSpecs.
        # ----------------------------------------------------

        # ----------------------------------------------------
        # Global gradient norm
        # ----------------------------------------------------

        grad_norm = optax.global_norm(
            grads
        )

        # ----------------------------------------------------
        # Clip AFTER reductions
        # ----------------------------------------------------

        clip_scale = jnp.minimum(
            1.0,
            max_grad_norm
            / jnp.maximum(
                grad_norm,
                1e-8,
            ),
        )

        grads = jax.tree_util.tree_map(
            lambda g: g * clip_scale,
            grads,
        )

        # ----------------------------------------------------
        # Optimizer update
        # ----------------------------------------------------

        updates, new_opt_state = (
            optimizer.update(
                grads,
                opt_state,
                params,
            )
        )

        new_params = optax.apply_updates(
            params,
            updates,
        )

        # ----------------------------------------------------
        # GLOBAL token accounting
        # ----------------------------------------------------

        tokens_in_step = (
            batch.shape[0]
            * batch.shape[1]
            * batch.shape[2]
        )

        # ----------------------------------------------------
        # State update
        # ----------------------------------------------------

        new_state = state.apply_grad_step(
            params=new_params,
            opt_state=new_opt_state,
            tokens_in_step=tokens_in_step,
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
    # Mesh-aware compilation
    # ========================================================

    with (
        jax.set_mesh(mesh),
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
                None,
            ),

            donate_argnums=(0,),
        )


# ============================================================
# Eval Step
# ============================================================

def create_eval_step(
    *,
    model,
    config,
    mesh,
    data_sharding,
):
    """
    Create GSPMD evaluation step.
    """

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
            inputs,
            mode="train",
        )

        logits = constrain_logits(
            logits
        )

        loss, metrics = compute_loss(
            logits,
            targets,
        )

        return {
            "loss": loss.astype(
                jnp.float32
            ),
        }

    with (
        jax.set_mesh(mesh),
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

            out_shardings=None,
        )
