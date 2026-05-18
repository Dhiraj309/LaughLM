"""LaughLM/distributed/state.py

Mesh-native abstract/sharded state utilities.

Design goals
------------
- zero-allocation abstract init
- mesh-native parameter initialization
- deterministic logical partition extraction
- GSPMD-safe sharded initialization
- scan/remat compatibility
- optimizer-state compatibility
- checkpoint-safe structure
"""

from __future__ import annotations

from typing import Any

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


# ============================================================
# Dummy inputs
# ============================================================

def create_dummy_inputs(
    input_shape,
):
    """
    Create abstract token inputs.

    Parameters
    ----------
    input_shape:
        [batch, sequence]
    """

    return jax.ShapeDtypeStruct(
        shape=input_shape,
        dtype=jnp.int32,
    )


# ============================================================
# Abstract state
# ============================================================

def create_abstract_state(
    *,
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Create abstract initialized state.

    Returns
    -------
    abstract_state
    logical_specs
    shardings

    Notes
    -----
    Uses:
    - jax.eval_shape
    - logical partition extraction
    - logical -> NamedSharding conversion
    """

    dummy_inputs = create_dummy_inputs(
        input_shape
    )

    def init_fn():

        return model.init(
            rng,
            input_ids=dummy_inputs,
            use_cache=False,
            mode="train",
        )

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        # ----------------------------------------------------
        # Zero-allocation abstract initialization
        # ----------------------------------------------------

        abstract_state = jax.eval_shape(
            init_fn
        )

        # ----------------------------------------------------
        # Extract logical PartitionSpecs
        # ----------------------------------------------------

        logical_specs = (
            nn.get_partition_spec(
                abstract_state
            )
        )

        # ----------------------------------------------------
        # Convert logical -> NamedSharding
        # ----------------------------------------------------

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


# ============================================================
# Sharded initialization
# ============================================================

def create_sharded_state(
    *,
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Materialize sharded initialized state.

    IMPORTANT
    ---------
    Avoids:
    - replicated host initialization
    - giant host memory spikes
    - post-init resharding

    This is the canonical TPU-safe path.
    """

    dummy_inputs = create_dummy_inputs(
        input_shape
    )

    def init_fn():

        return model.init(
            rng,
            input_ids=dummy_inputs,
            use_cache=False,
            mode="train",
        )

    (
        _abstract_state,
        _logical_specs,
        shardings,
    ) = create_abstract_state(
        model=model,
        config=config,
        mesh=mesh,
        rng=rng,
        input_shape=input_shape,
    )

    with (
        mesh,
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        # ----------------------------------------------------
        # Mesh-native sharded initialization
        # ----------------------------------------------------

        sharded_init_fn = jax.jit(
            init_fn,
            out_shardings=shardings,
        )

        state = sharded_init_fn()

    return state
