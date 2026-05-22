"""
LaughLM/distributed/sharding.py

PMAP-safe sharding compatibility helpers.

This branch uses PMAP as the production training backend.

Important:
- Runtime PMAP training must not depend on NamedSharding, PartitionSpec,
  mesh context, or with_sharding_constraint.
- The LLaMA model still imports logical constraint helpers from this file.
  For PMAP they intentionally behave as no-ops.
- Mesh/GSPMD helper functions are kept for future research branches, but
  they are not used by the PMAP trainer.
"""

from __future__ import annotations

from typing import Any


# ============================================================
# PMAP-safe logical constraints
# ============================================================

def constrain_batch(batch):
    return batch


def constrain_hidden_states(hidden_states):
    return hidden_states


def constrain_attention_tensor(tensor):
    return tensor


def constrain_kv_cache(tensor):
    return tensor


def constrain_logits(logits):
    return logits


def constrain_loss_tensor(tensor):
    return tensor


def shard_data(data: Any, sharding=None):
    return data


# ============================================================
# Future GSPMD compatibility helpers
# ============================================================

def _get_axis_names(config):
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

    return {
        "batch": "data",
        "embed": None,
        "heads": None,
        "kv_heads": None,
        "mlp": None,
        "vocab": None,
        "sequence": None,
        "layers": None,
    }


def get_logical_axis_rules(config):
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


def logical_to_sharding(logical_annotations, mesh, config):
    from flax.linen import partitioning as nn_partitioning
    import flax.linen as nn

    with nn_partitioning.axis_rules(get_logical_axis_rules(config)):
        return nn.logical_to_mesh_sharding(logical_annotations, mesh)


def create_named_sharding(mesh, *axes):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(mesh, P(*axes))


def replicated_sharding(mesh):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(mesh, P())


def create_input_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)
    return NamedSharding(mesh, P(None, axes["batch"], axes["sequence"]))


def create_token_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)
    return NamedSharding(mesh, P(axes["batch"], axes["sequence"]))


def create_activation_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)
    return NamedSharding(mesh, P(axes["batch"], axes["sequence"], axes["embed"]))


def create_logits_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)
    return NamedSharding(mesh, P(axes["batch"], axes["sequence"], axes["vocab"]))