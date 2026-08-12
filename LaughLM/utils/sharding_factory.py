"""
LaughLM/utils/sharding_factory.py

MaxText 3D Mesh Construction, Sequence Parallelism (SP) Specs & Activation Rematerialization.

Features:
1. Contiguous Submesh Construction: Enforces contiguous_submeshes=True in
   jax.experimental.mesh_utils to guarantee optimal 2D torus ICI interconnect
   routing on TPU v5e topologies.
2. Sequence Parallelism (SP): MaxText-style SP specs to partition activation layers
   along the sequence dimension (T) across tensor-parallel axes, eliminating LayerNorm/RMSNorm memory spikes.
3. Selective Activation Rematerialization: jax.ad.checkpoint policy wrappers
   (e.g., dots_saveable) to keep heavy attention matrices in SRAM while recomputing
   lightweight pointwise operations during backpropagation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Callable

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
import flax.linen as nn

from LaughLM.distributed.mesh import create_mesh as create_native_mesh
from LaughLM.config.schema import LaughLMConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Contiguous Submesh Mesh Construction
# ------------------------------------------------------------

def build_mesh(config: LaughLMConfig) -> Mesh:
    """
    Construct physical TPU mesh based on config.optimizations.sharding_strategy.

    Enforces contiguous_submeshes=True for optimal ICI interconnect routing.
    """
    sharding_strategy = getattr(
        getattr(config, "optimizations", None),
        "sharding_strategy",
        "fsdp",
    )

    devices = np.asarray(jax.devices())
    num_devices = devices.size

    if sharding_strategy == "maxtext_3d":
        mesh_cfg = config.spmd.mesh
        axis_sizes = mesh_cfg.axis_sizes()

        # Build 3D mesh shape: (data, fsdp, tensor)
        data_size = max(1, axis_sizes.get("data", 1))
        fsdp_size = max(1, axis_sizes.get("fsdp", 1))
        tensor_size = max(1, axis_sizes.get("tensor", 1))
        seq_size = max(1, axis_sizes.get("sequence", 1))

        # Align active axes for total devices
        if data_size * fsdp_size * tensor_size * seq_size != num_devices:
            # Fallback to automated 3D axis shape matching available devices
            if num_devices >= 8:
                fsdp_size = max(1, num_devices // (data_size * tensor_size))
            else:
                data_size = num_devices
                fsdp_size = 1
                tensor_size = 1

        active_shape = (data_size, fsdp_size, tensor_size)
        active_axes = ("data", "fsdp", "tensor")

        # Strip size-1 axes
        real_shape = tuple(s for s in active_shape if s > 1)
        real_names = tuple(name for s, name in zip(active_shape, active_axes) if s > 1)

        if not real_shape:
            real_shape = (1,)
            real_names = ("data",)

        device_mesh = mesh_utils.create_device_mesh(
            real_shape,
            devices=devices,
            contiguous_submeshes=True,
        )

        mesh = Mesh(device_mesh, axis_names=real_names)
        logger.info(
            f"[sharding_factory] Created MaxText 3D mesh: axis_names={mesh.axis_names}, shape={device_mesh.shape}"
        )
        return mesh

    # Default FSDP native mesh construction (which already sets contiguous_submeshes=True)
    return create_native_mesh(config)


# ------------------------------------------------------------
# Sequence Parallelism (SP) Partition Specs
# ------------------------------------------------------------

def get_sequence_parallel_spec(
    mesh: Mesh,
    axis_name: str = "tensor",
) -> P:
    """
    Generate MaxText-style Sequence Parallelism (SP) PartitionSpec.

    Partitions activations along sequence dimension (T) across tensor/sequence parallel axes.
    """
    active_axes = set(mesh.axis_names)

    if axis_name in active_axes and "data" in active_axes:
        # 2D Sequence Parallel spec: Batch sharded on data, Seq sharded on tensor
        return P("data", axis_name)
    elif axis_name in active_axes:
        return P(axis_name, None)
    elif "fsdp" in active_axes:
        # FSDP sequence sharding fallback
        return P("fsdp", None)
    elif "data" in active_axes:
        return P("data", None)

    return P(None, None)


def annotate_sequence_parallelism(
    x: jnp.ndarray,
    mesh: Mesh,
    axis_name: str = "tensor",
) -> jnp.ndarray:
    """
    Apply SP sharding constraint to layer activation tensors (e.g. RMSNorm / LayerNorm outputs).
    """
    spec = get_sequence_parallel_spec(mesh, axis_name=axis_name)
    sharding = NamedSharding(mesh, spec)
    return jax.lax.with_sharding_constraint(x, sharding)


# ------------------------------------------------------------
# Selective Activation Rematerialization Policy
# ------------------------------------------------------------

def get_remat_policy(config: LaughLMConfig) -> Callable:
    """
    Return jax.ad.checkpoint policy for activation rematerialization.

    'dots_saveable' keeps heavy matrix multiplications (attention/linear dots) in SRAM
    while recomputing lightweight pointwise operations (activations, norms) during backprop.
    """
    policy_name = getattr(config.spmd.remat, "policy", "dots_saveable")

    if policy_name == "dots_saveable":
        return jax.checkpoint_policies.dots_saveable
    elif policy_name == "dots_with_no_batch_dims_saveable":
        return jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    elif policy_name == "everything_saveable":
        return jax.checkpoint_policies.everything_saveable
    elif policy_name == "nothing_saveable":
        return jax.checkpoint_policies.nothing_saveable

    # Default to dots_saveable
    return jax.checkpoint_policies.dots_saveable


def apply_rematerialization(
    fn_or_module: Any,
    policy: Optional[Callable] = None,
    prevent_cse: bool = False,
) -> Any:
    """
    Wrap target function or module with jax.ad.checkpoint rematerialization.
    """
    if policy is None:
        policy = jax.checkpoint_policies.dots_saveable

    return jax.checkpoint(
        fn_or_module,
        policy=policy,
        prevent_cse=prevent_cse,
    )
