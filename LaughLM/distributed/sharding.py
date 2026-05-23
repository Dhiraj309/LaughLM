"""
LaughLM/distributed/sharding.py

Backend-aware logical sharding helpers.

PMAP:
  constraints are no-ops.

GSPMD/FSDP:
  constraints use Flax logical axes and are resolved by axis_rules.
"""

from __future__ import annotations

from typing import Any


_GSPMD_CONSTRAINTS_ENABLED = False


def enable_gspmd_constraints(enabled: bool = True):
    global _GSPMD_CONSTRAINTS_ENABLED
    _GSPMD_CONSTRAINTS_ENABLED = bool(enabled)


def gspmd_constraints_enabled() -> bool:
    return _GSPMD_CONSTRAINTS_ENABLED


def _logical_constraint(x, axes):
    if not _GSPMD_CONSTRAINTS_ENABLED:
        return x

    import flax.linen as nn

    return nn.with_logical_constraint(x, axes)


# ============================================================
# Logical activation constraints
# ============================================================

def constrain_batch(batch):
    # Token batch: [batch, sequence] or [grad_accum, batch, sequence]
    if getattr(batch, "ndim", None) == 3:
        return _logical_constraint(batch, (None, "batch", "sequence"))

    if getattr(batch, "ndim", None) == 2:
        return _logical_constraint(batch, ("batch", "sequence"))

    return batch


def constrain_hidden_states(hidden_states):
    return _logical_constraint(
        hidden_states,
        ("batch", "sequence", "embed"),
    )


def constrain_attention_tensor(tensor):
    return _logical_constraint(
        tensor,
        ("batch", "sequence", "heads", None),
    )


def constrain_attention_q(tensor):
    return _logical_constraint(
        tensor,
        ("batch", "sequence", "heads", None),
    )


def constrain_attention_kv(tensor):
    return _logical_constraint(
        tensor,
        ("batch", "sequence", "kv_heads", None),
    )


def constrain_kv_cache(tensor):
    return _logical_constraint(
        tensor,
        ("batch", "sequence", "kv_heads", None),
    )


def constrain_logits(logits):
    return _logical_constraint(
        logits,
        ("batch", "sequence", "vocab"),
    )


def constrain_loss_tensor(tensor):
    return _logical_constraint(
        tensor,
        ("batch", "sequence"),
    )


def shard_data(data: Any, sharding=None):
    if sharding is None:
        return data

    import jax

    return jax.device_put(data, sharding)


# ============================================================
# Logical axis helpers
# ============================================================

def _get_axis_names(config):
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

    return NamedSharding(
        mesh,
        P(
            None,
            axes["batch"],
            axes["sequence"],
        ),
    )


def create_token_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
        ),
    )


def create_activation_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
            axes["embed"],
        ),
    )


def create_logits_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(config)

    return NamedSharding(
        mesh,
        P(
            axes["batch"],
            axes["sequence"],
            axes["vocab"],
        ),
    )