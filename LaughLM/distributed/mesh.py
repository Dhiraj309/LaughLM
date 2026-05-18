"""
LaughLM/distributed/mesh.py

Frontier-grade device mesh utilities.
"""

from __future__ import annotations

import numpy as np
import jax

from jax.sharding import Mesh
from jax.experimental import mesh_utils


# ─────────────────────────────────────────────────────────────
# Device mesh
# ─────────────────────────────────────────────────────────────

def create_device_mesh(config):
    """
    Create physical device mesh.

    Returns
    -------
    np.ndarray
        Device mesh array.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    mesh_shape = [
        axis_sizes["data"],
        axis_sizes["fsdp"],
        axis_sizes["tensor"],
        axis_sizes["sequence"],
        axis_sizes["pipeline"],
    ]

    active_shape = [
        x for x in mesh_shape
        if x > 1
    ]

    devices = jax.devices()

    required_devices = 1

    for size in active_shape:
        required_devices *= size

    available_devices = len(devices)

    if required_devices != available_devices:

        raise ValueError(
            "Mesh/device mismatch:\n"
            f"  required_devices={required_devices}\n"
            f"  available_devices={available_devices}\n"
            f"  mesh_shape={mesh_shape}"
        )

    # --------------------------------------------------------
    # Single-device fallback
    # --------------------------------------------------------

    if len(active_shape) == 0:

        return np.asarray(devices).reshape((1,))

    return mesh_utils.create_device_mesh(
        active_shape,
        devices,
    )


# ─────────────────────────────────────────────────────────────
# Named mesh
# ─────────────────────────────────────────────────────────────

def create_mesh(config):
    """
    Create named JAX mesh.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    axis_names = [
        axis_name
        for axis_name, size
        in axis_sizes.items()
        if size > 1
    ]

    device_mesh = create_device_mesh(
        config
    )

    # --------------------------------------------------------
    # Single-device fallback
    # --------------------------------------------------------

    if len(axis_names) == 0:

        axis_names = ("data",)

    return Mesh(
        device_mesh,
        axis_names=tuple(axis_names),
    )
