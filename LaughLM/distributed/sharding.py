"""
LaughLM/distributed/sharding.py

Logical axis + NamedSharding utilities.

Frontier-grade SPMD helpers:
────────────────────────────────────────────
1. Logical axis rules
2. Logical → physical sharding conversion
3. Explicit NamedSharding helpers
4. Input batch sharding
5. Activation constraints
6. Loss/logit constraints
7. Future-ready sequence parallel hooks

References:
────────────────────────────────────────────
- MaxText
- T5X
- Levanter
- Pax
"""

from flax.linen import (
    partitioning as nn_partitioning,
)

from jax.sharding import (
    NamedSharding,
    PartitionSpec as P,
)

import flax.linen as nn


# ─────────────────────────────────────────────────────────────
# Logical axis rules
# ─────────────────────────────────────────────────────────────

def get_logical_axis_rules(config):
    """
    Convert config axis rules into Flax format.

    Logical axes:
    ─────────────────────────────────────────
    batch
    embed
    heads
    kv_heads
    mlp
    vocab
    sequence
    layers
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


# ─────────────────────────────────────────────────────────────
# Logical → physical sharding conversion
# ─────────────────────────────────────────────────────────────

def logical_to_sharding(
    logical_annotations,
    mesh,
    config,
):
    """
    Convert logical PartitionSpec
    tree into NamedSharding tree.
    """

    return nn.logical_to_mesh_sharding(
        logical_annotations,
        mesh,
        get_logical_axis_rules(config),
    )


# ─────────────────────────────────────────────────────────────
# Explicit NamedSharding helper
# ─────────────────────────────────────────────────────────────

def create_named_sharding(
    mesh,
    *axes,
):
    """
    Convenience helper.

    Example
    -------
    create_named_sharding(
        mesh,
        "data",
        "tensor",
    )
    """

    return NamedSharding(
        mesh,
        P(*axes),
    )


# ─────────────────────────────────────────────────────────────
# Input sharding
# ─────────────────────────────────────────────────────────────

def create_input_sharding(
    mesh,
):
    """
    Create canonical input batch sharding.

    Input batch shape:
    ─────────────────────────────────────────
    [grad_accum, global_batch, sequence]

    Logical layout:
    ─────────────────────────────────────────
    grad_accum -> replicated
    global_batch -> batch/data axis
    sequence -> replicated

    Result:
    ─────────────────────────────────────────
    PartitionSpec(
        None,
        "data",
        None,
    )
    """

    return NamedSharding(
        mesh,
        P(
            None,
            "data",
            None,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Batch constraints
# ─────────────────────────────────────────────────────────────

def constrain_batch(batch):
    """
    Apply logical constraints to input batch.

    Batch shape:
    ─────────────────────────────────────────
    [grad_accum, batch, sequence]
    """

    return nn_partitioning.with_logical_constraint(
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

def constrain_hidden_states(hidden_states):
    """
    Apply canonical hidden-state constraints.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, embed]
    """

    return nn_partitioning.with_logical_constraint(
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
    """
    Constraint for attention tensors.

    Shape:
    ─────────────────────────────────────────
    [batch, heads, sequence, head_dim]
    """

    return nn_partitioning.with_logical_constraint(
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
    """
    Constraint KV cache tensors.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, kv_heads, head_dim]
    """

    return nn_partitioning.with_logical_constraint(
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
    """
    Apply logits constraints.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, vocab]
    """

    return nn_partitioning.with_logical_constraint(
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
    """
    Constraint scalar-per-token losses.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence]
    """

    return nn_partitioning.with_logical_constraint(
        tensor,
        (
            "batch",
            "sequence",
        ),
    )
