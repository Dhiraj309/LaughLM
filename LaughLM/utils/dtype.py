import jax
import jax.numpy as jnp


def get_dtype(dtype_str: str):
    backend = jax.default_backend()

    # TPU → use bf16
    if backend == "tpu":
        if dtype_str == "bfloat16":
            return jnp.bfloat16

    # GPU fallback → use fp16 instead of bf16
    if backend == "gpu":
        if dtype_str == "bfloat16":
            return jnp.float16

    if dtype_str == "float16":
        return jnp.float16

    if dtype_str == "float32":
        return jnp.float32

    raise ValueError(f"Unsupported dtype: {dtype_str}")
