"""
LaughLM/distributed/state.py

Mesh-native abstract/sharded state utilities.

Responsibilities
────────────────────────────────────────────
1. Abstract parameter/state creation
2. Logical partition extraction
3. Logical → physical sharding conversion
4. Mesh-native parameter initialization
5. Compile-safe sharded init
6. Optimizer-state compatibility
7. Future-ready scan/remat support

References
────────────────────────────────────────────
- MaxText
- T5X
- Pax
- Levanter
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
# Abstract state creation
# ─────────────────────────────────────────────────────────────

def create_abstract_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Create abstract initialized model state.

    Returns:
    ─────────────────────────────────────────
    abstract_state
    logical_specs
    shardings

    Notes
    ─────────────────────────────────────────
    - Uses eval_shape only (no allocation)
    - Produces logical partition specs
    - Converts specs into NamedShardings
    - Compatible with remat + scan
    """

    dummy_input = jax.ShapeDtypeStruct(
        input_shape,
        jnp.int32,
    )

    def init_fn():

        return model.init(
            rng,
            dummy_input,
        )

    with (
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):

        # --------------------------------------------------
        # Abstract initialization
        # --------------------------------------------------

        abstract_state = jax.eval_shape(
            init_fn
        )

        # --------------------------------------------------
        # Extract logical PartitionSpecs
        # --------------------------------------------------

        logical_specs = nn.get_partition_spec(
            abstract_state
        )

        # --------------------------------------------------
        # Convert to concrete NamedShardings
        # --------------------------------------------------

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
# Sharded parameter/state initialization
# ─────────────────────────────────────────────────────────────

def create_sharded_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Materialize sharded initialized state directly
    onto devices.

    IMPORTANT
    ─────────────────────────────────────────
    This avoids:
    - host-side replicated initialization
    - giant host memory spikes
    - post-init resharding
    - TPU HBM duplication

    Equivalent to:
    - MaxText abstract init flow
    - T5X partitioned initialization
    """

    dummy_input = jax.ShapeDtypeStruct(
        input_shape,
        jnp.int32,
    )

    def init_fn():

        return model.init(
            rng,
            dummy_input,
        )

    # ------------------------------------------------------
    # Get abstract shardings
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # Mesh-native sharded initialization
    # ------------------------------------------------------

    with (
        jax.set_mesh(mesh),
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
