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
8. Replicated scalar shardings
9. Canonical activation shardings

References:
────────────────────────────────────────────
- MaxText
- T5X
- Levanter
- Pax
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
    Convert logical PartitionSpec tree
    into NamedSharding tree.
    """

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
# Replicated sharding
# ─────────────────────────────────────────────────────────────

def replicated_sharding(
    mesh,
):
    """
    Fully replicated sharding.

    Used for:
    - scalar losses
    - metrics
    - RNG seeds
    - counters
    """

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
    """
    Create canonical input batch sharding.

    Input batch shape:
    ─────────────────────────────────────────
    [grad_accum, global_batch, sequence]

    Logical layout:
    ─────────────────────────────────────────
    grad_accum -> replicated
    global_batch -> batch/data axis
    sequence -> optional sequence axis
    """

    rules = config.spmd.axis_rules

    return NamedSharding(
        mesh,
        P(
            None,
            rules.batch,
            rules.sequence,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Token tensor sharding
# ─────────────────────────────────────────────────────────────

def create_token_sharding(
    mesh,
    config,
):
    """
    Standard token tensor sharding.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence]
    """

    rules = config.spmd.axis_rules

    return NamedSharding(
        mesh,
        P(
            rules.batch,
            rules.sequence,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Hidden-state sharding
# ─────────────────────────────────────────────────────────────

def create_activation_sharding(
    mesh,
    config,
):
    """
    Standard hidden-state sharding.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, embed]
    """

    rules = config.spmd.axis_rules

    return NamedSharding(
        mesh,
        P(
            rules.batch,
            rules.sequence,
            rules.embed,
        ),
    )


# ─────────────────────────────────────────────────────────────
# Logits sharding
# ─────────────────────────────────────────────────────────────

def create_logits_sharding(
    mesh,
    config,
):
    """
    Standard logits sharding.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, vocab]
    """

    rules = config.spmd.axis_rules

    return NamedSharding(
        mesh,
        P(
            rules.batch,
            rules.sequence,
            rules.vocab,
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
    """
    Apply canonical hidden-state constraints.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, embed]
    """

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
    """
    Constraint for attention tensors.

    Shape:
    ─────────────────────────────────────────
    [batch, heads, sequence, head_dim]
    """

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
    """
    Constraint KV cache tensors.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, kv_heads, head_dim]
    """

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
    """
    Apply logits constraints.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence, vocab]
    """

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
    """
    Constraint scalar-per-token losses.

    Shape:
    ─────────────────────────────────────────
    [batch, sequence]
    """

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
    """
    Explicitly place data onto mesh.

    Useful for:
    - input batches
    - eval batches
    - token tensors
    """

    return nn_partitioning.with_sharding_constraint(
        data,
        sharding.spec,
    )