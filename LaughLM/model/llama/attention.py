"""
LaughLM/model/llama/attention.py

Canonical LLaMA attention for:
- PMAP production
- TPU Splash under PMAP
- TPU Splash under GSPMD via shard_map

Backend policy:
- standard / xla / flash / cudnn / memory_efficient -> XLA SDPA
- splash + PMAP  -> TPU SplashAttention directly
- splash + GSPMD -> TPU SplashAttention wrapped with shard_map
- decode -> XLA SDPA

Important fix:
- SplashAttention expects caller-side Q/K scaling semantics.
- HF/JAX scaled dot-product attention uses scale = 1 / sqrt(head_dim).
- Therefore canonical Splash path scales Q by 1 / sqrt(head_dim)
  before calling the Splash kernel.

Legacy compatibility:
- Old checkpoints trained before this fix used unscaled Splash logits.
- To reproduce old behavior temporarily, set:
    LAUGHLM_SPLASH_LEGACY_UNSCALED=1
  or, during old-checkpoint export:
    LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH=1
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import jax
import jax.numpy as jnp

try:
    from jax.sharding import PartitionSpec as P
except Exception:
    P = None

from flax import linen as nn

from LaughLM.model.llama.config import LlamaConfig

from LaughLM.model.llama.initialization import (
    create_dense,
    constrain_hidden_states,
)

from LaughLM.model.llama.rope import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
)

from LaughLM.model.llama.kv_cache import (
    KVCache,
    update_kv_cache,
)

from LaughLM.distributed.sharding import (
    constrain_kv_cache,
    gspmd_constraints_enabled,
    get_current_mesh,
)


_LOGGED_BACKENDS = set()
_VALID_FALLBACK_POLICIES = {"warn", "error"}


def _log_attention_backend(name: str):
    if name not in _LOGGED_BACKENDS:
        _LOGGED_BACKENDS.add(name)
        print(f"[attention] using {name}", flush=True)


def _is_tpu_backend() -> bool:
    return jax.default_backend() == "tpu"


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "0")

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _legacy_unscaled_splash_enabled() -> bool:
    """
    Compatibility mode for old checkpoints trained with unscaled Splash logits.

    Use only for validating/exporting old checkpoints.

    Canonical future behavior should leave this disabled.
    """

    return (
        _truthy_env("LAUGHLM_SPLASH_LEGACY_UNSCALED")
        or _truthy_env("LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH")
    )


def _scale_query_for_splash(
    query_states: jnp.ndarray,
) -> jnp.ndarray:
    """
    Canonical scaled dot-product attention semantics.

    HF/JAX attention computes:

        softmax((Q @ K^T) / sqrt(head_dim)) @ V

    Splash kernel path receives Q/K/V directly, so scale Q here:

        Q_scaled = Q / sqrt(head_dim)

    Legacy mode disables this for old checkpoints trained before this fix.
    """

    if _legacy_unscaled_splash_enabled():
        _log_attention_backend(
            "legacy unscaled splash attention"
        )
        return query_states

    head_dim = query_states.shape[-1]

    scale = jnp.asarray(
        head_dim ** -0.5,
        dtype=query_states.dtype,
    )

    return query_states * scale


def _attention_fallback_policy(config: LlamaConfig) -> str:
    """
    Fallback policy resolution order:
    1. config.attention_fallback, if wired through config_factory
    2. LAUGHLM_ATTENTION_FALLBACK env var
    3. "warn"
    """

    policy = getattr(config, "attention_fallback", None)

    if policy is None:
        policy = os.environ.get(
            "LAUGHLM_ATTENTION_FALLBACK",
            "warn",
        )

    policy = str(policy).lower()

    if policy not in _VALID_FALLBACK_POLICIES:
        warnings.warn(
            "[attention] Invalid attention_fallback="
            f"{policy!r}; using 'warn'. "
            f"Valid values: {sorted(_VALID_FALLBACK_POLICIES)}",
            RuntimeWarning,
        )
        return "warn"

    return policy


def _handle_splash_fallback(
    *,
    config: LlamaConfig,
    reason: str,
    exc: Exception | None = None,
):
    policy = _attention_fallback_policy(config)

    message = (
        "[attention] Splash fallback requested but "
        f"attention_fallback='{policy}'. Reason: {reason}"
    )

    if policy == "error":
        raise RuntimeError(message) from exc

    warnings.warn(
        message + "; falling back to XLA SDPA.",
        RuntimeWarning,
    )


def _find_splash_block_size(
    seq_len: int,
    requested_block: int,
) -> tuple[int, int]:
    supported_blocks = (128, 256, 512, 1024)
    if requested_block not in supported_blocks:
        raise ValueError(
            f"Unsupported SplashAttention block size: {requested_block}. "
            f"Expected one of {supported_blocks}."
        )

    if seq_len % requested_block == 0:
        return requested_block, 0

    pad = (
        ((seq_len + requested_block - 1) // requested_block)
        * requested_block
        - seq_len
    )

    return requested_block, pad


def _pad_for_splash(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    *,
    block_size: int = 512,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """
    q/k/v layout:
        [B, T, H, Dh]
    """

    B, T, QH, Dh = q.shape
    KVH = k.shape[2]

    _, pad = _find_splash_block_size(T, block_size)

    if pad == 0:
        return q, k, v, 0

    q = jnp.concatenate(
        [
            q,
            jnp.zeros(
                (B, pad, QH, Dh),
                dtype=q.dtype,
            ),
        ],
        axis=1,
    )

    k = jnp.concatenate(
        [
            k,
            jnp.zeros(
                (B, pad, KVH, Dh),
                dtype=k.dtype,
            ),
        ],
        axis=1,
    )

    v = jnp.concatenate(
        [
            v,
            jnp.zeros(
                (B, pad, KVH, Dh),
                dtype=v.dtype,
            ),
        ],
        axis=1,
    )

    return q, k, v, pad


def _splash_attention(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    *,
    block_size: int = 512,
) -> jnp.ndarray:
    """
    TPU SplashAttention.

    Input/output layout:
        [B, T, H, Dh]

    Splash kernel layout:
        per-example [H, T, Dh]

    Canonical behavior:
        Q is scaled by 1 / sqrt(head_dim) before entering Splash,
        so Splash matches XLA/HF scaled dot-product attention semantics.
    """

    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )

    query_states = _scale_query_for_splash(
        query_states
    )

    q, k, v, pad_amount = _pad_for_splash(
        query_states,
        key_states,
        value_states,
        block_size=block_size,
    )

    B, T, QH, Dh = q.shape
    KVH = k.shape[2]

    if QH != KVH:
        raise NotImplementedError(
            "Splash path currently requires "
            "num_attention_heads == num_key_value_heads. "
            "Use XLA SDPA for true GQA/MQA."
        )

    block, _ = _find_splash_block_size(T, block_size)

    _log_attention_backend(
        f"splash attention block={block} seq={T}"
    )

    # [B, T, H, Dh] -> [B, H, T, Dh]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    causal_mask = splash_attention_mask.CausalMask(
        shape=(T, T)
    )

    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(causal_mask,) * QH
    )

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block,
        block_kv=block,
        block_kv_compute=block,
        block_q_dkv=block,
        block_kv_dkv=block,
        block_kv_dkv_compute=block,
        block_q_dq=block,
        block_kv_dq=block,
    )

    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        block_sizes=block_sizes,
        head_shards=1,
        q_seq_shards=1,
    )

    def per_example(q_b, k_b, v_b):
        return splash_kernel(
            q_b,
            k_b,
            v_b,
            None,
        )

    out = jax.vmap(
        per_example,
        in_axes=(0, 0, 0),
    )(q, k, v)

    # [B, H, T, Dh] -> [B, T, H, Dh]
    out = jnp.transpose(
        out,
        (0, 2, 1, 3),
    )

    if pad_amount > 0:
        out = out[:, :-pad_amount, :, :]

    return out


def _splash_attention_shard_map(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    *,
    block_size: int = 512,
) -> jnp.ndarray:
    """
    GSPMD-compatible TPU SplashAttention.

    Pallas/Mosaic kernels cannot be automatically partitioned by GSPMD.
    Therefore we wrap Splash with shard_map.

    Expected global layout:
        q/k/v: [batch, sequence, heads, head_dim]

    Current safe sharding:
        batch -> data
        sequence/head/head_dim replicated inside each local shard
        FSDP axis is not used by the local Splash kernel
    """

    mesh = get_current_mesh()

    if mesh is None:
        raise RuntimeError(
            "GSPMD Splash requires current mesh to be registered. "
            "Call set_current_mesh(mesh) in FSDPTrainer after mesh creation."
        )

    if "data" not in mesh.axis_names:
        raise RuntimeError(
            "GSPMD Splash shard_map currently requires a 'data' mesh axis. "
            "Use hybrid mesh such as data=2, fsdp=4. "
            "For pure fsdp=8, use attention_impl='standard'."
        )

    if P is None:
        raise RuntimeError(
            "jax.sharding.PartitionSpec is unavailable."
        )

    spec = P("data", None, None, None)

    def local_splash(q, k, v):
        return _splash_attention(
            q,
            k,
            v,
            block_size=block_size,
        )

    try:
        mapped = jax.shard_map(
            local_splash,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )

    except AttributeError:
        from jax.experimental.shard_map import shard_map

        mapped = shard_map(
            local_splash,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_rep=False,
        )

    except TypeError:
        from jax.experimental.shard_map import shard_map

        mapped = shard_map(
            local_splash,
            mesh=mesh,
            in_specs=(spec, spec, spec),
            out_specs=spec,
            check_rep=False,
        )

    _log_attention_backend(
        "gspmd shard_map splash attention"
    )

    return mapped(
        query_states,
        key_states,
        value_states,
    )


def _attention_impl_from_config(
    config: LlamaConfig,
    mode: str,
    q_len: int,
    kv_len: int,
) -> str:
    impl = getattr(
        config,
        "attention_impl",
        "standard",
    )

    if mode == "decode":
        return "xla"

    if impl == "splash":
        if (
            _is_tpu_backend()
            and q_len == kv_len
            and q_len > 4
        ):
            return "splash"

        reason = (
            "attention_impl='splash' requested, but runtime is not "
            f"eligible for Splash. backend={jax.default_backend()!r}, "
            f"q_len={q_len}, kv_len={kv_len}, mode={mode!r}"
        )

        _handle_splash_fallback(
            config=config,
            reason=reason,
        )

        return "xla"

    return "xla"


def _xla_sdpa(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    mode: str,
) -> jnp.ndarray:
    """
    XLA SDPA fallback.

    Explicitly uses scale = 1 / sqrt(head_dim), matching HF LLaMA
    and canonical scaled dot-product attention.
    """

    q_len = query_states.shape[1]
    kv_len = key_states.shape[1]

    use_causal_flag = (
        attention_mask is None
        and mode in ("train", "prefill")
        and q_len == kv_len
    )

    if use_causal_flag:
        _log_attention_backend(
            "xla dot_product_attention causal"
        )
    else:
        _log_attention_backend(
            "xla dot_product_attention"
        )

    head_dim = query_states.shape[-1]

    scale = head_dim ** -0.5

    return jax.nn.dot_product_attention(
        query_states,
        key_states,
        value_states,
        bias=attention_mask,
        is_causal=use_causal_flag,
        scale=scale,
    )


from LaughLM.utils.fused_ops import fused_attention

def _attention(
    query_states: jnp.ndarray,
    key_states: jnp.ndarray,
    value_states: jnp.ndarray,
    attention_mask: Optional[jnp.ndarray],
    config: LlamaConfig,
    mode: str,
) -> jnp.ndarray:
    kernel_backend = getattr(config.optimizations, "kernel_backend", "native") if config.optimizations else "native"
    if kernel_backend == "tokamax":
        return fused_attention(
            query_states,
            key_states,
            value_states,
            mask=attention_mask,
            scale=config.head_dim ** -0.5,
            kernel_backend="tokamax",
        )

    backend = _attention_impl_from_config(
        config=config,
        mode=mode,
        q_len=query_states.shape[1],
        kv_len=key_states.shape[1],
    )

    if backend == "splash":
        try:
            if gspmd_constraints_enabled():
                return _splash_attention_shard_map(
                    query_states,
                    key_states,
                    value_states,
                    block_size=config.splash_block_size,
                )

            return _splash_attention(
                query_states,
                key_states,
                value_states,
                block_size=config.splash_block_size,
            )

        except Exception as e:
            _handle_splash_fallback(
                config=config,
                reason=f"{type(e).__name__}: {e}",
                exc=e,
            )

    return _xla_sdpa(
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
        mode=mode,
    )


class LlamaAttention(nn.Module):
    config: LlamaConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        positions: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        kv_cache: Optional[KVCache] = None,
        mode: str = "train",
    ) -> tuple[jnp.ndarray, Optional[KVCache]]:

        hidden_states = constrain_hidden_states(
            hidden_states
        )

        config = self.config

        B, T, _ = hidden_states.shape

        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim

        if getattr(config, "fused_qkv", False):
            qkv_proj = create_dense(
                features=q_dim + 2 * kv_dim,
                config=config,
                use_bias=config.attention_bias,
                name="qkv_proj",
            )

            qkv_states = qkv_proj(
                hidden_states
            )

            query_states = qkv_states[..., :q_dim]

            key_states = qkv_states[
                ...,
                q_dim : q_dim + kv_dim,
            ]

            value_states = qkv_states[
                ...,
                q_dim + kv_dim :,
            ]

        else:
            q_proj = create_dense(
                features=q_dim,
                config=config,
                use_bias=config.attention_bias,
                name="q_proj",
            )

            k_proj = create_dense(
                features=kv_dim,
                config=config,
                use_bias=config.attention_bias,
                name="k_proj",
            )

            v_proj = create_dense(
                features=kv_dim,
                config=config,
                use_bias=config.attention_bias,
                name="v_proj",
            )

            query_states = q_proj(
                hidden_states
            )

            key_states = k_proj(
                hidden_states
            )

            value_states = v_proj(
                hidden_states
            )

        o_proj = create_dense(
            features=config.hidden_size,
            config=config,
            use_bias=config.attention_bias,
            name="o_proj",
        )

        query_states = query_states.reshape(
            B,
            T,
            num_heads,
            head_dim,
        )

        key_states = key_states.reshape(
            B,
            T,
            num_kv_heads,
            head_dim,
        )

        value_states = value_states.reshape(
            B,
            T,
            num_kv_heads,
            head_dim,
        )

        rotary_emb = RotaryEmbedding(
            config
        )

        cos, sin = rotary_emb(
            query_states,
            positions,
        )

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
        )

        key_states = constrain_kv_cache(
            key_states
        )

        value_states = constrain_kv_cache(
            value_states
        )

        updated_cache = None

        if kv_cache is not None:
            (
                updated_cache,
                key_states,
                value_states,
            ) = update_kv_cache(
                kv_cache,
                key_states,
                value_states,
            )

            kv_length = updated_cache.cache_position

            key_states = key_states[
                :,
                :kv_length,
                :,
                :,
            ]

            value_states = value_states[
                :,
                :kv_length,
                :,
                :,
            ]

        query_states = query_states.astype(
            config.compute_dtype
        )

        key_states = key_states.astype(
            config.compute_dtype
        )

        value_states = value_states.astype(
            config.compute_dtype
        )

        attn_output = _attention(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            config=config,
            mode=mode,
        )

        attn_output = attn_output.reshape(
            B,
            T,
            config.hidden_size,
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        attn_output = o_proj(
            attn_output
        )

        attn_output = constrain_hidden_states(
            attn_output
        )

        return attn_output, updated_cache
