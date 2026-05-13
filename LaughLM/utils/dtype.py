"""
LaughLM/utils/dtype.py

Dtype resolution for LaughLM.

Frontier-grade changes (perf/frontier-optim):
──────────────────────────────────────────────
1. REMOVED the silent bf16 → fp16 fallback on GPU.
   All modern GPUs (T4, V100, A100, H100, L4, L40S) support bfloat16.
   The old code silently downgraded bf16 → fp16 on GPU, which:
   - Hurts training stability (fp16 has only 5 exponent bits vs bf16's 8)
   - Requires loss scaling (not implemented in this codebase)
   - Produces different results on GPU vs TPU with the same config

2. ADDED get_dtype_from_config() — reads from the new SPMDConfig.dtype
   block instead of the legacy parallelism config. Returns a tuple of
   (param_dtype, compute_dtype, output_dtype) as jnp dtypes.

3. ADDED resolve_compute_dtype() and resolve_param_dtype() convenience
   functions for downstream modules that only need one dtype.

Reference: MaxText uses bfloat16 on both TPU and GPU identically.
"""

import jax.numpy as jnp
from typing import Tuple


# ────────────────────────────────────────────────────────────────
# String → jnp.dtype mapping
# ────────────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "float32":  jnp.float32,
    "float16":  jnp.float16,
    "bfloat16": jnp.bfloat16,
}


def get_dtype(dtype_str: str) -> jnp.dtype:
    """
    Convert a dtype string to a JAX dtype.

    No backend-dependent fallbacks — the config says what it means.
    If you want bfloat16, you get bfloat16 on every backend.

    Parameters
    ----------
    dtype_str : "float32", "float16", or "bfloat16"

    Returns
    -------
    jnp.dtype

    Raises
    ------
    ValueError
        If dtype_str is not a recognized dtype.
    """
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: '{dtype_str}'. "
            f"Valid options: {list(_DTYPE_MAP.keys())}"
        )
    return dtype


# ────────────────────────────────────────────────────────────────
# Config-aware dtype resolution (reads from SPMDConfig.dtype)
# ────────────────────────────────────────────────────────────────

def get_dtype_from_config(config) -> Tuple[jnp.dtype, jnp.dtype, jnp.dtype]:
    """
    Resolve all three dtypes from config.spmd.dtype.

    Returns
    -------
    (param_dtype, compute_dtype, output_dtype) as jnp dtypes.

    Usage
    -----
        param_dt, compute_dt, output_dt = get_dtype_from_config(config)
    """
    dtype_cfg = config.spmd.dtype
    return (
        get_dtype(dtype_cfg.param_dtype),
        get_dtype(dtype_cfg.compute_dtype),
        get_dtype(dtype_cfg.output_dtype),
    )


def resolve_compute_dtype(config) -> jnp.dtype:
    """Shorthand: get just the compute dtype from config."""
    return get_dtype(config.spmd.dtype.compute_dtype)


def resolve_param_dtype(config) -> jnp.dtype:
    """Shorthand: get just the param dtype from config."""
    return get_dtype(config.spmd.dtype.param_dtype)


def resolve_output_dtype(config) -> jnp.dtype:
    """Shorthand: get just the output dtype from config."""
    return get_dtype(config.spmd.dtype.output_dtype)