"""
LaughLM/distributed/sharding.py

Backend-aware logical sharding helpers.

PMAP:
  constraints are no-ops.

GSPMD/FSDP:
  constraints use Flax logical axes and are resolved by axis_rules.

Phase 6.1 guardrail:
  JAX Mesh removes axes whose size is 1. Therefore helpers must not create
  NamedSharding specs that reference inactive/missing mesh axes.

Example:
  config axis_rules.batch = "data"
  mesh axis_names = ("fsdp",) because data=1
  P(None, "data", None) is invalid.

We sanitize missing mesh axes to None before constructing NamedSharding.
"""

from __future__ import annotations

from typing import Any


_GSPMD_CONSTRAINTS_ENABLED = False
_CURRENT_MESH = None


def enable_gspmd_constraints(enabled: bool = True):
    global _GSPMD_CONSTRAINTS_ENABLED
    _GSPMD_CONSTRAINTS_ENABLED = bool(enabled)


def gspmd_constraints_enabled() -> bool:
    return _GSPMD_CONSTRAINTS_ENABLED


def set_current_mesh(mesh):
    global _CURRENT_MESH
    _CURRENT_MESH = mesh


def get_current_mesh():
    return _CURRENT_MESH


def _logical_constraint(x, axes):
    if not _GSPMD_CONSTRAINTS_ENABLED:
        return x

    import flax.linen as nn

    return nn.with_logical_constraint(
        x,
        axes,
    )


# ============================================================
# Mesh-axis sanitization
# ============================================================

def _mesh_axis_names(mesh) -> set[str]:
    if mesh is None:
        return set()

    return set(
        getattr(
            mesh,
            "axis_names",
            (),
        )
    )


def sanitize_axis_for_mesh(axis, mesh):
    """
    Drop mesh axes that are not active in the JAX Mesh.

    JAX Mesh construction removes size-one axes. A config can still contain
    logical rules such as batch -> data, but if data=1 the actual mesh may
    not contain the "data" axis.

    Returns
    -------
    axis | None
        Same axis if valid, otherwise None.
    """

    if axis is None:
        return None

    axis_names = _mesh_axis_names(
        mesh
    )

    if isinstance(axis, str):
        if axis in axis_names:
            return axis

        return None

    if isinstance(axis, tuple):
        filtered = tuple(
            a
            for a in axis
            if a in axis_names
        )

        if len(filtered) == 0:
            return None

        return filtered

    return axis


def sanitize_axes_for_mesh(mesh, *axes):
    return tuple(
        sanitize_axis_for_mesh(
            axis,
            mesh,
        )
        for axis in axes
    )


# ============================================================
# Logical activation constraints
# ============================================================

def constrain_batch(batch):
    # Token batch: [batch, sequence] or [grad_accum, batch, sequence]
    if getattr(batch, "ndim", None) == 3:
        return _logical_constraint(
            batch,
            (None, "batch", "sequence"),
        )

    if getattr(batch, "ndim", None) == 2:
        return _logical_constraint(
            batch,
            ("batch", "sequence"),
        )

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

    return jax.device_put(
        data,
        sharding,
    )


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


def get_logical_axis_rules(config, mesh=None):
    """
    Return Flax logical axis rules.

    If mesh is provided, remove inactive physical mesh axes.

    This is important because LaughLM's mesh.py intentionally removes
    size-one axes from the actual JAX Mesh.
    """

    axes = _get_axis_names(
        config
    )

    if mesh is not None:
        axes = {
            name: sanitize_axis_for_mesh(
                axis,
                mesh,
            )
            for name, axis in axes.items()
        }

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

    with nn_partitioning.axis_rules(
        get_logical_axis_rules(
            config,
            mesh=mesh,
        )
    ):
        return nn.logical_to_mesh_sharding(
            logical_annotations,
            mesh,
        )


def create_named_sharding(mesh, *axes):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(
        mesh,
        P(
            *sanitize_axes_for_mesh(
                mesh,
                *axes,
            )
        ),
    )


def replicated_sharding(mesh):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    return NamedSharding(
        mesh,
        P(),
    )


def create_input_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(
        config
    )

    return NamedSharding(
        mesh,
        P(
            *sanitize_axes_for_mesh(
                mesh,
                None,
                axes["batch"],
                axes["sequence"],
            )
        ),
    )


def create_token_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(
        config
    )

    return NamedSharding(
        mesh,
        P(
            *sanitize_axes_for_mesh(
                mesh,
                axes["batch"],
                axes["sequence"],
            )
        ),
    )


def create_activation_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(
        config
    )

    return NamedSharding(
        mesh,
        P(
            *sanitize_axes_for_mesh(
                mesh,
                axes["batch"],
                axes["sequence"],
                axes["embed"],
            )
        ),
    )


def create_logits_sharding(mesh, config):
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    axes = _get_axis_names(
        config
    )

    return NamedSharding(
        mesh,
        P(
            *sanitize_axes_for_mesh(
                mesh,
                axes["batch"],
                axes["sequence"],
                axes["vocab"],
            )
        ),
    )
