"""
LaughLM/export/convert_params.py
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ============================================================
# Helpers
# ============================================================


def _to_numpy(x):
    """
    Robust conversion:
    Orbax/JAX/Sharded -> NumPy
    """

    import jax
    import numpy as np

    # ========================================================
    # Unwrap nested dict wrappers
    # ========================================================

    while isinstance(x, dict):

        # common Orbax wrapper patterns
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

            # fallback:
            # single-key dict unwrap
            if len(x) == 1:

                x = next(iter(x.values()))

            else:

                raise TypeError(
                    "Cannot unwrap dict.\n"
                    f"keys={list(x.keys())}"
                )

    # ========================================================
    # Materialize device arrays
    # ========================================================

    x = jax.device_get(x)

    arr = np.asarray(x)

    # ========================================================
    # Object wrapper handling
    # ========================================================

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

        # dict unwrap again
        while isinstance(x, dict):

            if len(x) == 1:

                x = next(iter(x.values()))

            else:

                raise TypeError(
                    "Nested dict wrapper unresolved.\n"
                    f"keys={list(x.keys())}"
                )

        arr = np.asarray(x)

    # ========================================================
    # bf16 -> fp32
    # ========================================================

    if str(arr.dtype) == "bfloat16":

        arr = arr.astype(np.float32)

    return arr
    

def _kernel_to_weight(kernel):

    return _to_numpy(kernel).T


def _has_key(tree, key):

    try:
        return key in tree

    except Exception:
        return False


def _is_mapping(x):

    return hasattr(x, "keys")


# ============================================================
# Layout detection
# ============================================================


def _is_scan_layout(params):

    if not _has_key(params, "model"):
        return False

    model = params["model"]

    if not _has_key(model, "layers"):
        return False

    layers = model["layers"]

    return _has_key(layers, "block")


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
    Never materializes tensors.
    Uses shape metadata only.
    """

    block = params["model"]["layers"]["block"]

    def slice_tree(tree):

        # ----------------------------------------------------
        # Recursive mapping
        # ----------------------------------------------------

        if _is_mapping(tree):

            return {
                k: slice_tree(v)
                for k, v in tree.items()
            }

        # ----------------------------------------------------
        # Non-array leaves
        # ----------------------------------------------------

        shape = getattr(tree, "shape", None)

        if shape is None:
            return tree

        # ----------------------------------------------------
        # Scalars
        # ----------------------------------------------------

        if len(shape) == 0:
            return tree

        # ----------------------------------------------------
        # Slice ONLY scanned tensors
        # ----------------------------------------------------

        if shape[0] == num_layers:

            return tree[layer_idx]

        return tree

    return slice_tree(block)


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


def _export_norm(
    tensors,
    flax_module,
    hf_prefix,
):

    tensors[
        f"{hf_prefix}.weight"
    ] = _to_numpy(
        flax_module["weight"]
    )


# ============================================================
# Layer conversion
# ============================================================


def _convert_attention(
    tensors,
    layer,
    prefix,
):

    attn = layer["self_attn"]

    _export_dense(
        tensors,
        attn["q_proj"],
        f"{prefix}.self_attn.q_proj",
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
):

    prefix = f"model.layers.{layer_idx}"

    _convert_attention(
        tensors,
        layer,
        prefix,
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

        if not isinstance(
            tensor,
            np.ndarray,
        ):

            raise TypeError(
                f"{name}: not ndarray"
            )

        if tensor.dtype == np.dtype("O"):

            raise TypeError(
                f"{name}: object dtype"
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

    print("[export] embeddings...")

    tensors[
        "model.embed_tokens.weight"
    ] = _to_numpy(
        model["embed_tokens"]["embedding"]
    )

    # ========================================================
    # Layers
    # ========================================================

    print("[export] layers...")

    if _is_scan_layout(params):

        num_layers = (
            config.num_hidden_layers
        )

        for layer_idx in range(num_layers):

            layer = _extract_scan_layer(
                params,
                layer_idx,
                num_layers,
            )

            _convert_layer(
                tensors,
                layer,
                layer_idx,
            )

    else:

        layers = _extract_non_scan_layers(
            params
        )

        for (
            layer_idx,
            layer,
        ) in enumerate(layers):

            _convert_layer(
                tensors,
                layer,
                layer_idx,
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

    if not config.tie_word_embeddings:

        print("[export] lm_head...")

        _export_dense(
            tensors,
            params["lm_head"],
            "lm_head",
        )

    return tensors
