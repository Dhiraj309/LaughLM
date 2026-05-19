"""
LaughLM/distributed/mesh.py

Frontier-grade device mesh utilities.

Frontier-grade fixes (2026):
────────────────────────────────────────────
1. TPU topology-aware mesh layouts
2. Contiguous submeshes for TPU performance
3. Multi-host safe mesh semantics
4. Deterministic logical-axis ordering
5. Proper active-axis extraction
6. Safe validation for FSDP layouts
7. Future tensor-parallel compatibility
"""

from __future__ import annotations

import numpy as np
import jax

from jax.sharding import Mesh
from jax.experimental import mesh_utils


# ============================================================
# Helpers
# ============================================================

_LOGICAL_AXIS_ORDER = (
    "data",
    "fsdp",
    "tensor",
    "sequence",
    "pipeline",
)


# ============================================================
# Device mesh
# ============================================================

def create_device_mesh(config):
    """
    Create physical device mesh.

    IMPORTANT
    ─────────────────────────────────────────
    Uses topology-aware mesh construction
    for TPU performance.

    Returns
    -------
    np.ndarray
        Physical device mesh.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    #
    # Deterministic logical ordering
    #

    mesh_shape = [
        axis_sizes[axis]
        for axis in _LOGICAL_AXIS_ORDER
    ]

    #
    # Remove inactive axes
    #

    active_shape = tuple(
        size
        for size in mesh_shape
        if size > 1
    )

    #
    # Physical devices
    #

    devices = np.asarray(
        jax.devices()
    )

    available_devices = (
        devices.size
    )

    #
    # Required device count
    #

    required_devices = 1

    for size in active_shape:
        required_devices *= size

    #
    # Single-device fallback
    #

    if len(active_shape) == 0:

        if available_devices != 1:

            raise ValueError(
                "Single-device mesh requested "
                f"but found {available_devices} devices"
            )

        return devices.reshape((1,))

    #
    # Validation
    #

    if required_devices != available_devices:

        raise ValueError(
            "Mesh/device mismatch:\n"
            f"  required_devices={required_devices}\n"
            f"  available_devices={available_devices}\n"
            f"  mesh_shape={mesh_shape}\n"
            f"  active_shape={active_shape}"
        )

    #
    # IMPORTANT
    #
    # contiguous_submeshes=True
    #
    # Critical for:
    # - TPU v4/v5e/v5p
    # - multi-host slices
    # - FSDP collectives
    # - tensor-parallel locality
    #

    device_mesh = (
        mesh_utils.create_device_mesh(
            active_shape,
            devices=devices,
            contiguous_submeshes=True,
        )
    )

    return device_mesh


# ============================================================
# Named mesh
# ============================================================

def create_mesh(config):
    """
    Create named JAX mesh.

    Returns
    -------
    Mesh
        Named GSPMD mesh.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    #
    # Keep axis ordering deterministic
    #

    axis_names = tuple(
        axis
        for axis in _LOGICAL_AXIS_ORDER
        if axis_sizes[axis] > 1
    )

    device_mesh = create_device_mesh(
        config
    )

    #
    # Single-device fallback
    #

    if len(axis_names) == 0:

        axis_names = ("data",)

    #
    # Final validation
    #

    if len(axis_names) != len(
        device_mesh.shape
    ):

        raise ValueError(
            "Axis/device mesh rank mismatch:\n"
            f"  axis_names={axis_names}\n"
            f"  device_mesh.shape={device_mesh.shape}"
        )

    mesh = Mesh(
        device_mesh,
        axis_names=axis_names,
    )

    print(
        "[mesh] created:\n"
        f"  axis_names={mesh.axis_names}\n"
        f"  shape={mesh.devices.shape}"
    )

    return mesh
