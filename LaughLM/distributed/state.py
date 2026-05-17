import jax
import flax.linen as nn

from flax.linen import partitioning as nn_partitioning

from LaughLM.distributed.sharding import (
    get_logical_axis_rules,
    logical_to_sharding,
)


def create_abstract_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Create abstract sharded state.
    """

    dummy = jax.ShapeDtypeStruct(
        input_shape,
        jax.numpy.int32,
    )

    def init_fn():
        return model.init(
            rng,
            dummy,
        )

    with (
        jax.set_mesh(mesh),
        nn_partitioning.axis_rules(
            get_logical_axis_rules(config)
        ),
    ):
        abstract_state = jax.eval_shape(
            init_fn
        )

        logical_specs = nn.get_partition_spec(
            abstract_state
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


def create_sharded_state(
    model,
    config,
    mesh,
    rng,
    input_shape,
):
    """
    Materialize sharded parameters directly on devices.
    """

    dummy = jax.ShapeDtypeStruct(
        input_shape,
        jax.numpy.int32,
    )

    def init_fn():
        return model.init(
            rng,
            dummy,
        )

    (
        _,
        _,
        shardings,
    ) = create_abstract_state(
        model,
        config,
        mesh,
        rng,
        input_shape,
    )

    with jax.set_mesh(mesh):

        sharded_init = jax.jit(
            init_fn,
            out_shardings=shardings,
        )

        return sharded_init()
