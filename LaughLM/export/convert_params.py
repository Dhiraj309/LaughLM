"""
LaughLM/export/convert_params.py

Convert native LaughLM/Flax LLaMA parameters to Hugging Face LlamaForCausalLM
safetensors-compatible parameter names and layouts.

Important layout rules:
- Flax Dense kernel:      [in_features, out_features]
- PyTorch Linear weight: [out_features, in_features]
- Therefore every Dense kernel is transposed during export.

Important fused QKV rule:
- LaughLM native fused QKV layout is contiguous:
    [Q_all_heads, K_all_heads, V_all_heads]
- This matches LaughLM/model/llama/attention.py:
    query_states = qkv_states[..., :q_dim]
    key_states   = qkv_states[..., q_dim : q_dim + kv_dim]
    value_states = qkv_states[..., q_dim + kv_dim :]

Legacy unscaled Splash export:
- Old checkpoints trained before Splash Q scaling used effectively unscaled
  attention logits:
    logits = Q @ K.T
- HF LLaMA uses:
    logits = Q @ K.T / sqrt(head_dim)
- To emulate the old checkpoint in HF, scale exported q_proj by sqrt(head_dim).
- Enable only for old checkpoints:
    LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH=1
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict

import numpy as np


# ============================================================
# Helpers
# ============================================================


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "0")

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _legacy_unscaled_splash_export_enabled() -> bool:
    """
    Compatibility mode for checkpoints trained before Splash Q scaling
    was fixed.

    Use only for old checkpoints:
        LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH=1
    """

    return _truthy_env(
        "LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH"
    )


_LEGACY_Q_SCALE_WARNED = False


def _legacy_q_scale(config) -> float:
    return math.sqrt(
        float(config.head_dim)
    )


def _maybe_scale_q_kernel_for_legacy_unscaled_splash(
    q_kernel: np.ndarray,
    config,
) -> np.ndarray:
    """
    If legacy export is enabled, multiply q_proj kernel by sqrt(head_dim).

    HF applies / sqrt(head_dim) inside attention.
    This export-time scaling makes HF emulate old unscaled Splash logits.
    """

    global _LEGACY_Q_SCALE_WARNED

    if not _legacy_unscaled_splash_export_enabled():
        return q_kernel

    scale = _legacy_q_scale(
        config
    )

    if not _LEGACY_Q_SCALE_WARNED:
        print(
            "[export] WARNING: "
            "LAUGHLM_EXPORT_LEGACY_UNSCALED_SPLASH=1 enabled. "
            "Scaling exported q_proj by "
            f"sqrt(head_dim)={scale:.6f}. "
            "Use this only for old checkpoints trained before "
            "canonical Splash Q scaling was fixed."
        )

        _LEGACY_Q_SCALE_WARNED = True

    return q_kernel * scale


def _maybe_scale_q_bias_for_legacy_unscaled_splash(
    q_bias: np.ndarray,
    config,
) -> np.ndarray:
    """
    Same legacy compatibility scaling for q_proj bias.
    """

    if not _legacy_unscaled_splash_export_enabled():
        return q_bias

    scale = _legacy_q_scale(
        config
    )

    return q_bias * scale


def _to_numpy(x):
    """
    Robust conversion:
    Orbax/JAX/Sharded/LogicallyPartitioned-ish wrappers -> NumPy.
    """

    import jax
    import numpy as np

    while isinstance(x, dict):
        for key in (
            "value",
            "array",
            "data",
            "tensor",
            "embedding",
            "kernel",
            "weight",
        ):
            if key in x:
                x = x[key]
                break
        else:
            if len(x) == 1:
                x = next(iter(x.values()))
            else:
                raise TypeError(
                    "Cannot unwrap dict.\n"
                    f"keys={list(x.keys())}"
                )

    x = jax.device_get(x)

    arr = np.asarray(
        x
    )

    while arr.dtype == object:
        if arr.shape == ():
            x = arr.item()

        elif arr.size == 1:
            x = arr.reshape(-1)[0]

        else:
            raise TypeError(
                "Unresolved object array.\n"
                f"shape={arr.shape}"
            )

        while isinstance(x, dict):
            if len(x) == 1:
                x = next(iter(x.values()))

            else:
                raise TypeError(
                    "Nested dict wrapper unresolved.\n"
                    f"keys={list(x.keys())}"
                )

        arr = np.asarray(
            x
        )

    if str(arr.dtype) == "bfloat16":
        arr = arr.astype(
            np.float32
        )

    return arr


def _kernel_to_weight(kernel):
    """
    Flax Dense kernel [in, out] -> PyTorch Linear weight [out, in].
    """

    return _to_numpy(kernel).T


def _has_key(tree, key):
    try:
        return key in tree
    except Exception:
        return False


def _is_mapping(x):
    return hasattr(
        x,
        "keys",
    )


# ============================================================
# Layout detection
# ============================================================


def _is_scan_layout(params):
    """
    Detect scanned layer layout:

        params["model"]["layers"]["block"]

    Non-scanned layout uses:

        params["model"]["layers_0"]
        params["model"]["layers_1"]
        ...
    """

    if not _has_key(params, "model"):
        return False

    model = params["model"]

    if not _has_key(model, "layers"):
        return False

    layers = model["layers"]

    return _has_key(
        layers,
        "block",
    )


# ============================================================
# Scan extraction
# ============================================================


def _extract_scan_layer(
    params,
    layer_idx,
    num_layers,
):
    """
    Extract one layer from scan layout.

    IMPORTANT:
    This slices only leaves whose first dimension equals num_layers.
    """

    block = params["model"]["layers"]["block"]

    def slice_tree(tree):
        if _is_mapping(tree):
            return {
                k: slice_tree(v)
                for k, v in tree.items()
            }

        shape = getattr(
            tree,
            "shape",
            None,
        )

        if shape is None:
            return tree

        if len(shape) == 0:
            return tree

        if shape[0] == num_layers:
            return tree[layer_idx]

        return tree

    return slice_tree(
        block
    )


# ============================================================
# Non-scan extraction
# ============================================================


def _extract_non_scan_layers(params):
    model = params["model"]

    layer_keys = sorted(
        [
            k
            for k in model.keys()
            if k.startswith("layers_")
        ],
        key=lambda x: int(
            x.split("_")[-1]
        ),
    )

    return [
        model[k]
        for k in layer_keys
    ]


# ============================================================
# Export helpers
# ============================================================


def _export_dense(
    tensors,
    flax_module,
    hf_prefix,
):
    tensors[
        f"{hf_prefix}.weight"
    ] = _kernel_to_weight(
        flax_module["kernel"]
    )

    if _has_key(flax_module, "bias"):
        tensors[
            f"{hf_prefix}.bias"
        ] = _to_numpy(
            flax_module["bias"]
        )


def _export_q_dense(
    tensors,
    flax_module,
    hf_prefix,
    config,
):
    """
    Export q_proj, with optional legacy unscaled-Splash compatibility scaling.
    """

    q_kernel = _to_numpy(
        flax_module["kernel"]
    )

    q_kernel = _maybe_scale_q_kernel_for_legacy_unscaled_splash(
        q_kernel,
        config,
    )

    tensors[
        f"{hf_prefix}.weight"
    ] = q_kernel.T

    if _has_key(flax_module, "bias"):
        q_bias = _to_numpy(
            flax_module["bias"]
        )

        q_bias = _maybe_scale_q_bias_for_legacy_unscaled_splash(
            q_bias,
            config,
        )

        tensors[
            f"{hf_prefix}.bias"
        ] = q_bias


def _export_norm(
    tensors,
    flax_module,
    hf_prefix,
):
    """
    RMSNorm weights are 1D scale vectors, no transpose.
    LaughLM RMSNorm stores the scale as "weight".
    HF LlamaRMSNorm expects "<prefix>.weight".
    """

    tensors[
        f"{hf_prefix}.weight"
    ] = _to_numpy(
        flax_module["weight"]
    )


def _split_fused_qkv_kernel(
    kernel: np.ndarray,
    config,
):
    """
    Split LaughLM fused QKV kernel using native contiguous layout:

        [Q_all_heads, K_all_heads, V_all_heads]

    Returns Flax-layout kernels:
        q_kernel: [hidden_size, q_dim]
        k_kernel: [hidden_size, kv_dim]
        v_kernel: [hidden_size, kv_dim]
    """

    hidden_size = int(
        config.hidden_size
    )

    num_heads = int(
        config.num_attention_heads
    )

    num_kv_heads = int(
        config.num_key_value_heads
    )

    head_dim = int(
        config.head_dim
    )

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    expected_total = q_dim + 2 * kv_dim

    if kernel.ndim != 2:
        raise ValueError(
            "Expected fused QKV kernel rank 2.\n"
            f"got shape={kernel.shape}"
        )

    expected_shape = (
        hidden_size,
        expected_total,
    )

    if kernel.shape != expected_shape:
        raise ValueError(
            "Bad fused QKV kernel shape.\n"
            f"got={kernel.shape}\n"
            f"expected={expected_shape}\n"
            f"hidden_size={hidden_size}\n"
            f"num_heads={num_heads}\n"
            f"num_kv_heads={num_kv_heads}\n"
            f"head_dim={head_dim}\n"
            f"q_dim={q_dim}\n"
            f"kv_dim={kv_dim}"
        )

    q_kernel = kernel[:, :q_dim]

    k_kernel = kernel[
        :,
        q_dim : q_dim + kv_dim,
    ]

    v_kernel = kernel[
        :,
        q_dim + kv_dim :,
    ]

    return q_kernel, k_kernel, v_kernel


def _split_fused_qkv_bias(
    bias: np.ndarray,
    config,
):
    """
    Split LaughLM fused QKV bias using native contiguous layout:

        [Q_all_heads, K_all_heads, V_all_heads]
    """

    num_heads = int(
        config.num_attention_heads
    )

    num_kv_heads = int(
        config.num_key_value_heads
    )

    head_dim = int(
        config.head_dim
    )

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    expected_total = q_dim + 2 * kv_dim

    expected_shape = (
        expected_total,
    )

    if bias.shape != expected_shape:
        raise ValueError(
            "Bad fused QKV bias shape.\n"
            f"got={bias.shape}\n"
            f"expected={expected_shape}\n"
            f"q_dim={q_dim}\n"
            f"kv_dim={kv_dim}"
        )

    q_bias = bias[:q_dim]

    k_bias = bias[
        q_dim : q_dim + kv_dim
    ]

    v_bias = bias[
        q_dim + kv_dim :
    ]

    return q_bias, k_bias, v_bias


# ============================================================
# Layer conversion
# ============================================================


def _convert_attention(
    tensors,
    layer,
    prefix,
    config,
):
    attn = layer["self_attn"]

    if _has_key(attn, "qkv_proj"):
        qkv = attn["qkv_proj"]

        kernel = _to_numpy(
            qkv["kernel"]
        )

        (
            q_kernel,
            k_kernel,
            v_kernel,
        ) = _split_fused_qkv_kernel(
            kernel,
            config,
        )

        q_kernel = _maybe_scale_q_kernel_for_legacy_unscaled_splash(
            q_kernel,
            config,
        )

        tensors[
            f"{prefix}.self_attn.q_proj.weight"
        ] = q_kernel.T

        tensors[
            f"{prefix}.self_attn.k_proj.weight"
        ] = k_kernel.T

        tensors[
            f"{prefix}.self_attn.v_proj.weight"
        ] = v_kernel.T

        if _has_key(qkv, "bias"):
            bias = _to_numpy(
                qkv["bias"]
            )

            (
                q_bias,
                k_bias,
                v_bias,
            ) = _split_fused_qkv_bias(
                bias,
                config,
            )

            q_bias = _maybe_scale_q_bias_for_legacy_unscaled_splash(
                q_bias,
                config,
            )

            tensors[
                f"{prefix}.self_attn.q_proj.bias"
            ] = q_bias

            tensors[
                f"{prefix}.self_attn.k_proj.bias"
            ] = k_bias

            tensors[
                f"{prefix}.self_attn.v_proj.bias"
            ] = v_bias

    else:
        _export_q_dense(
            tensors,
            attn["q_proj"],
            f"{prefix}.self_attn.q_proj",
            config,
        )

        _export_dense(
            tensors,
            attn["k_proj"],
            f"{prefix}.self_attn.k_proj",
        )

        _export_dense(
            tensors,
            attn["v_proj"],
            f"{prefix}.self_attn.v_proj",
        )

    _export_dense(
        tensors,
        attn["o_proj"],
        f"{prefix}.self_attn.o_proj",
    )


def _convert_mlp(
    tensors,
    layer,
    prefix,
):
    mlp = layer["mlp"]

    _export_dense(
        tensors,
        mlp["gate_proj"],
        f"{prefix}.mlp.gate_proj",
    )

    _export_dense(
        tensors,
        mlp["up_proj"],
        f"{prefix}.mlp.up_proj",
    )

    _export_dense(
        tensors,
        mlp["down_proj"],
        f"{prefix}.mlp.down_proj",
    )


def _convert_norms(
    tensors,
    layer,
    prefix,
):
    _export_norm(
        tensors,
        layer["input_layernorm"],
        f"{prefix}.input_layernorm",
    )

    _export_norm(
        tensors,
        layer["post_attention_layernorm"],
        f"{prefix}.post_attention_layernorm",
    )


def _convert_layer(
    tensors,
    layer,
    layer_idx,
    config,
):
    prefix = f"model.layers.{layer_idx}"

    _convert_attention(
        tensors,
        layer,
        prefix,
        config,
    )

    _convert_mlp(
        tensors,
        layer,
        prefix,
    )

    _convert_norms(
        tensors,
        layer,
        prefix,
    )


# ============================================================
# Validation
# ============================================================


def validate_exported_tensors(
    tensors,
):
    print("[export] validating tensors...")

    total_params = 0

    for name, tensor in tensors.items():
        if not isinstance(tensor, np.ndarray):
            raise TypeError(
                f"{name}: not ndarray; got={type(tensor)}"
            )

        if tensor.dtype == np.dtype("O"):
            raise TypeError(
                f"{name}: object dtype"
            )

        if not np.all(np.isfinite(tensor)):
            raise ValueError(
                f"{name}: contains NaN or Inf"
            )

        total_params += tensor.size

    print(
        f"[export] validated "
        f"{len(tensors):,} tensors"
    )

    print(
        f"[export] total params: "
        f"{total_params:,}"
    )


# ============================================================
# Public API
# ============================================================


def convert_params_to_hf(
    params,
    config,
) -> Dict[str, np.ndarray]:
    tensors = {}

    model = params["model"]

    # ========================================================
    # Embeddings
    # ========================================================

    print(
        "[export] embeddings: materializing host array...",
        flush=True,
    )
    embedding_start = time.perf_counter()
    embedding_weight = _to_numpy(
        model["embed_tokens"]["embedding"]
    )
    print(
        "[export] embeddings: ready "
        f"shape={embedding_weight.shape} "
        f"dtype={embedding_weight.dtype} "
        f"elapsed={time.perf_counter() - embedding_start:.2f}s",
        flush=True,
    )
    tensors[
        "model.embed_tokens.weight"
    ] = embedding_weight

    # ========================================================
    # Layers
    # ========================================================

    print("[export] layers...")

    if _is_scan_layout(params):
        num_layers = int(
            config.num_hidden_layers
        )

        for layer_idx in range(
            num_layers
        ):
            layer = _extract_scan_layer(
                params,
                layer_idx,
                num_layers,
            )

            _convert_layer(
                tensors,
                layer,
                layer_idx,
                config,
            )

    else:
        layers = _extract_non_scan_layers(
            params
        )

        if len(layers) != int(config.num_hidden_layers):
            raise ValueError(
                "Layer count mismatch.\n"
                f"found={len(layers)}\n"
                f"expected={config.num_hidden_layers}"
            )

        for layer_idx, layer in enumerate(
            layers
        ):
            _convert_layer(
                tensors,
                layer,
                layer_idx,
                config,
            )

    # ========================================================
    # Final norm
    # ========================================================

    print("[export] final norm...")

    _export_norm(
        tensors,
        model["norm"],
        "model.norm",
    )

    # ========================================================
    # LM head
    # ========================================================

    if config.tie_word_embeddings:
        print(
            "[export] tied lm_head: relying on HF tie_word_embeddings=True"
        )

    else:
        print("[export] lm_head...")

        _export_dense(
            tensors,
            params["lm_head"],
            "lm_head",
        )

    return tensors
