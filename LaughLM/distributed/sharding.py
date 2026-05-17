from flax.linen import partitioning as nn_partitioning

from jax.sharding import (
    NamedSharding,
    PartitionSpec as P,
)

import flax.linen as nn


def get_logical_axis_rules(config):
    """
    Convert config axis rules into Flax format.
    """

    rules = config.spmd.axis_rules

    return (
        ("batch", rules.batch),
        ("embed", rules.embed),
        ("heads", rules.heads),
        ("kv_heads", rules.kv_heads),
        ("mlp", rules.mlp),
        ("vocab", rules.vocab),
        ("sequence", rules.sequence),
        ("layers", rules.layers),
    )


def logical_to_sharding(
    logical_annotations,
    mesh,
    config,
):
    """
    Convert logical PartitionSpec
    to NamedSharding.
    """

    return nn.logical_to_mesh_sharding(
        logical_annotations,
        mesh,
        get_logical_axis_rules(config),
    )


def create_named_sharding(
    mesh,
    *axes,
):
    """
    Convenience helper.
    """

    return NamedSharding(
        mesh,
        P(*axes),
    )
