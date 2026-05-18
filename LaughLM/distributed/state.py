"""
LaughLM/distributed/state.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import flax.linen as nn

from flax.linen import (
    partitioning as nn_partitioning,
)

from LaughLM.distributed.sharding import (
    get_logical_axis_rules,
    logical_to_sharding,
)


# ─────────────────────────────────────────────────────────────
# Dummy inputs
# ─────────────────────────────────────────────────────────────

def create_dummy_inputs(
    input_shape,
):
    """
    Create abstract token inputs.
    """

    input_ids = jax.ShapeDtypeStruct(
        input_shape,
        jnp.int32,
    )

    return input_ids


# ─────────────────────────────────────────────────────────────
# Abstract state
# ─────────────────────────────────────────────────────────────

def create_abstract_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):

    dummy_inputs = create_dummy_inputs(
        input_shape
    )

    def init_fn():

        return model.init(
            rng,
            dummy_inputs,
        )

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        abstract_state = jax.eval_shape(
            init_fn
        )

        logical_specs = (
            nn.get_partition_spec(
                abstract_state
            )
        )

        shardings = logical_to_sharding(
            logical_specs,
            mesh,
            config,
        )

    return (
        abstract_state,
        logical_specs,
        shardings,
    )


# ─────────────────────────────────────────────────────────────
# Sharded init
# ─────────────────────────────────────────────────────────────

def create_sharded_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):

    dummy_inputs = create_dummy_inputs(
        input_shape
    )

    def init_fn():

        return model.init(
            rng,
            dummy_inputs,
        )

    (
        _abstract_state,
        _logical_specs,
        shardings,
    ) = create_abstract_state(
        model,
        config,
        mesh,
        rng,
        input_shape,
    )

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        sharded_init_fn = jax.jit(
            init_fn,
            out_shardings=shardings,
        )

        state = sharded_init_fn()

    return state
