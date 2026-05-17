import numpy as np
import jax

from jax.sharding import Mesh
from jax.experimental import mesh_utils


def create_device_mesh(config):
    """
    Create physical device mesh.

    Returns
    -------
    np.ndarray of devices shaped according
    to active mesh axes.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    active_axes = [
        axis_sizes["data"],
        axis_sizes["fsdp"],
        axis_sizes["tensor"],
        axis_sizes["sequence"],
        axis_sizes["pipeline"],
    ]

    active_axes = [
        x for x in active_axes
        if x > 1
    ]

    devices = jax.devices()

    mesh = mesh_utils.create_device_mesh(
        active_axes,
        devices,
    )

    return mesh


def create_mesh(config):
    """
    Create named JAX mesh.
    """

    mesh_cfg = config.spmd.mesh

    axis_sizes = mesh_cfg.axis_sizes()

    axis_names = [
        k for k, v in axis_sizes.items()
        if v > 1
    ]

    device_mesh = create_device_mesh(config)

    return Mesh(
        device_mesh,
        axis_names=tuple(axis_names),
    )
