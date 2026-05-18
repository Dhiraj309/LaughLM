"""
LaughLM/distributed/sharding.py
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn

from flax.linen import (
    partitioning as nn_partitioning,
    with_logical_constraint,
)

from jax.sharding import (
    NamedSharding,
    PartitionSpec as P,
)


# ─────────────────────────────────────────────────────────────
# Axis helpers
# ─────────────────────────────────────────────────────────────

def _get_axis_names(config):
    """
    Unified axis extraction.

    Supports:
    - legacy global config
    - standalone LlamaConfig
    """

    #
    # Legacy config path
    #

    if hasattr(config, "spmd"):

        rules = config.spmd.axis_rules

        return {
            "batch": rules.batch,
            "embed": rules.embed,
            "heads": rules.heads,
            "kv_heads": rules.kv_heads,
            "mlp": rules.mlp,
            "vocab": rules.vocab,
            "sequence": rules.sequence,
            "layers": rules.layers,
        }

    #
    # Standalone LlamaConfig path
    #
    # Current TPU runtime:
    # 1D batch/data parallel only
    #

    return {
        "batch": "batch",
        "embed": None,
        "heads": None,
        "kv_heads": None,
        "mlp": None,
        "vocab": None,
        "sequence": None,
        "layers": None,
    }


# ─────────────────────────────────────────────────────────────
# Logical axis rules
# ─────────────────────────────────────────────────────────────

def get_logical_axis_rules(config):
    """
    Convert logical axis rules
    into Flax axis_rules format.
    """

    axes = _get_axis_names(config)

    return (
        ("batch", axes["batch"]),
        ("embed", axes["embed"]),
        ("heads", axes["heads"]),
        ("kv_heads", axes["kv_heads"]),
        ("mlp", axes["mlp"]),
        ("vocab", axes["vocab"]),
        ("sequence", axes["sequence"]),
        ("layers", axes["layers"]),
    )


# ─────────────────────────────────────────────────────────────
# Logical → physical sharding conversion
# ─────────────────────────────────────────────────────────────

def logical_to_sharding(
    logical_annotations,
    mesh,
    config,
):

    with nn_partitioning.axis_rules(
        get_logical_axis_rules(config)
    ):

        return nn.logical_to_mesh_sharding(
            logical_annotations,
            mesh,
        )


# ─────────────────────────────────────────────────────────────
# Explicit NamedSharding helper
# ─────────────────────────────────────────────────────────────

def create_named_sharding(
    mesh,
    *axes,
):

    return NamedSharding(
        mesh,
        P(*axes),
    )


# ─────────────────────────────────────────────────────────────
# Replicated sharding
# ─────────────────────────────────────────────────────────────

def replicated_sharding(
    mesh,
):

    return NamedSharding(
        mesh,
        P(),
    )


# ─────────────────────────────────────────────────────────────
# Input sharding
# ─────────────────────────────────────────────────────────────

def create_input_sharding(
    mesh,
    config,
):

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            None,
            axes["batch"],
            axes["sequence"],
        ),
    )


# ─────────────────────────────────────────────────────────────
# Token tensor sharding
# ─────────────────────────────────────────────────────────────

def create_token_sharding(
    mesh,
    config,
):

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
        ),
    )


# ─────────────────────────────────────────────────────────────
# Hidden-state sharding
# ─────────────────────────────────────────────────────────────

def create_activation_sharding(
    mesh,
    config,
):

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
            axes["embed"],
        ),
    )


# ─────────────────────────────────────────────────────────────
# Logits sharding
# ─────────────────────────────────────────────────────────────

def create_logits_sharding(
    mesh,
    config,
):

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
            axes["vocab"],
        ),
    )


# ─────────────────────────────────────────────────────────────
# Batch constraints
# ─────────────────────────────────────────────────────────────

def constrain_batch(batch):

    return with_logical_constraint(
        batch,
        (
            None,
            "batch",
            "sequence",
        ),
    )


# ─────────────────────────────────────────────────────────────
# Hidden-state constraints
# ─────────────────────────────────────────────────────────────

def constrain_hidden_states(
    hidden_states,
):

    return with_logical_constraint(
        hidden_states,
        (
            "batch",
            "sequence",
            "embed",
        ),
    )


# ─────────────────────────────────────────────────────────────
# Attention constraints
# ─────────────────────────────────────────────────────────────

def constrain_attention_tensor(
    tensor,
):

    return with_logical_constraint(
        tensor,
        (
            "batch",
            "heads",
            "sequence",
            None,
        ),
    )


# ─────────────────────────────────────────────────────────────
# KV cache constraints
# ─────────────────────────────────────────────────────────────

def constrain_kv_cache(
    tensor,
):

    return with_logical_constraint(
        tensor,
        (
            "batch",
            "sequence",
            "kv_heads",
            None,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Logits constraints
# ─────────────────────────────────────────────────────────────

def constrain_logits(
    logits,
):

    return with_logical_constraint(
        logits,
        (
            "batch",
            "sequence",
            "vocab",
        ),
    )


# ─────────────────────────────────────────────────────────────
# Loss constraints
# ─────────────────────────────────────────────────────────────

def constrain_loss_tensor(
    tensor,
):

    return with_logical_constraint(
        tensor,
        (
            "batch",
            "sequence",
        ),
    )


# ─────────────────────────────────────────────────────────────
# Explicit data placement
# ─────────────────────────────────────────────────────────────

def shard_data(
    data: Any,
    sharding,
):

    return nn_partitioning.with_sharding_constraint(
        data,
        sharding.spec,
    )
