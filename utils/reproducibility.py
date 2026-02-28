import hashlib
import json
from typing import Any

import jax
import jax.numpy as jnp

def create_master_key(seed: int) -> jax.random.PRNGKey:
    """
    Create root PRNG key from experiment seed.
    """
    return jax.random.PRNGKey(seed)

def fold_in_step(key: jax.random.PRNGKey, step: int) -> jax.random.PRNGKey:
    """
    Fold training step into RNG for deterministic per-step randomness.
    """
    return jax.random.fold_in(key, step)


def split_for_devices(key: jax.random.PRNGKey, num_devices: int):
    """
    Split key across devices deterministically.
    """
    return jax.random.split(key, num_devices)

def create_step_keys(
    master_key: jax.random.PRNGKey,
    step: int,
    num_devices: int,
):
    """
    Generate deterministic per-device step keys.
    """
    step_key = fold_in_step(master_key, step)
    device_keys = split_for_devices(step_key, num_devices)
    return device_keys

def hash_config(config: Any) -> str:
    """
    Deterministic hash of RootConfig.
    Used for checkpoint integrity + logging.
    """
    config_dict = _to_serializable(config)
    json_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def _to_serializable(obj):
    """
    Recursively convert dataclasses to dictionaries.
    """
    if hasattr(obj, "__dataclass_fields__"):
        return {
            field: _to_serializable(getattr(obj, field))
            for field in obj.__dataclass_fields__
        }
    elif isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    elif isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    else:
        return obj
