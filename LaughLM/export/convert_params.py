"""
LaughLM/export/convert_params.py

Convert LaughLM JAX/Flax parameters into
Hugging Face-compatible tensor mappings.

Design goals
------------
- scan-compatible
- tied-embedding aware
- safetensors-ready
- deterministic naming
- zero Torch dependency
- NumPy serialization output
- robust scan slicing
- FrozenDict-safe traversal
"""

from __future__ import annotations

from typing import Dict

import numpy as np


# ============================================================
# Helpers
# ============================================================


import jax
import numpy as np


def _to_numpy(x):
    """
    Convert JAX/Orbax arrays -> plain NumPy arrays.

    Handles:
    - jax.Array
    - GlobalDeviceArray
    - ShardedDeviceArray
    - Orbax-restored leaves
    - bf16 normalization
    """

    # --------------------------------------------------------
    # Materialize device array first
    # --------------------------------------------------------

    try:
        import jax

        x = jax.device_get(x)

    except Exception:
        pass

    # --------------------------------------------------------
    # Convert to ndarray
    # --------------------------------------------------------

    x = np.asarray(x)

    # --------------------------------------------------------
    # Orbax/JAX sometimes returns object arrays
    # wrapping a real ndarray scalar/object.
    # Unwrap recursively.
    # --------------------------------------------------------

    while x.dtype == np.dtype("O"):

        # scalar object wrapper
        if x.shape == ():

            x = x.item()
            x = np.asarray(x)
            continue

        # single-element object array
        if x.size == 1:

            x = np.asarray(
                x.reshape(-1)[0]
            )

            continue

        raise TypeError(
            "Object dtype encountered during export.\n"
            f"shape={x.shape}\n"
            f"dtype={x.dtype}"
        )

    # --------------------------------------------------------
    # bf16 -> fp32
    # --------------------------------------------------------

    if str(x.dtype) == "bfloat16":

        x = x.astype(np.float32)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not np.issubdtype(
        x.dtype,
        np.number,
    ):

        raise TypeError(
            f"Non-numeric dtype: {x.dtype}"
        )

    return x

def _kernel_to_weight(kernel):
    """
    Flax Dense kernel:
        [in_features, out_features]

    Torch Linear weight:
        [out_features, in_features]
    """

    return _to_numpy(kernel).T


def _has_key(tree, key):
    """
    FrozenDict-safe key check.
    """

    try:
        return key in tree

    except Exception:
        return False


def _is_mapping(x):
    """
    Generic mapping detection.

    Works for:
    - dict
    - FrozenDict
    - flax.core.FrozenDict
    """

    return hasattr(x, "keys")


# ============================================================
# Layout detection
# ============================================================


def _is_scan_layout(params):
    """
    Detect nn.scan parameter layout.

    Expected structure:

    params[
        "model"
    ][
        "layers"
    ][
        "block"
    ]
    """

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
    Extract one layer from scanned parameter tree.

    ONLY tensors whose first dimension equals
    num_layers are sliced.

    This prevents corruption of:
    - embeddings
    - norm weights
    - biases
    - non-scanned tensors
    """

    block = params["model"]["layers"]["block"]

    def slice_tree(tree):

        # ----------------------------------------------------
        # Recursive mappings
        # ----------------------------------------------------

        if _is_mapping(tree):

            return {
                k: slice_tree(v)
                for k, v in tree.items()
            }

        # ----------------------------------------------------
        # Non-array leaves
        # ----------------------------------------------------

        if not hasattr(tree, "shape"):

            return tree

        arr = np.asarray(tree)

        # ----------------------------------------------------
        # Scalars
        # ----------------------------------------------------

        if arr.ndim == 0:

            return tree

        # ----------------------------------------------------
        # Slice ONLY scanned tensors
        # ----------------------------------------------------

        if arr.shape[0] == num_layers:

            return tree[layer_idx]

        # ----------------------------------------------------
        # Non-scanned tensor
        # ----------------------------------------------------

        return tree

    return slice_tree(block)


# ============================================================
# Non-scan extraction
# ============================================================


def _extract_non_scan_layers(params):
    """
    Extract decoder layers from non-scanned layout.
    """

    model = params["model"]

    layer_keys = [
        k
        for k in model.keys()
        if k.startswith("layers_")
    ]

    layer_keys = sorted(
        layer_keys,
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
    """
    Export Dense layer.

    Supports:
    - kernel
    - optional bias
    """

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
    """
    Export RMSNorm.
    """

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
    """
    Convert attention tensors.
    """

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
    """
    Convert SwiGLU MLP tensors.
    """

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
    """
    Convert RMSNorm tensors.
    """

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
    """
    Convert one decoder layer.
    """

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
    """
    Validate exported tensors before safetensors save.
    """

    print(
        "[export] validating tensors..."
    )

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
            f"{name:<60}"
            f"{str(tensor.shape):<24}"
            f"{tensor.dtype}"
        )

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
    """
    Convert LaughLM params ->
    Hugging Face-compatible tensor dict.

    Returns
    -------
    Dict[str, np.ndarray]

    Compatible with:

        safetensors.numpy.save_file(...)
    """

    tensors = {}

    model = params["model"]

    # ========================================================
    # Embeddings
    # ========================================================

    embedding = model[
        "embed_tokens"
    ]["embedding"]

    tensors[
        "model.embed_tokens.weight"
    ] = _to_numpy(
        embedding
    )

    # ========================================================
    # Decoder layers
    # ========================================================

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
    # Final RMSNorm
    # ========================================================

    _export_norm(
        tensors,
        model["norm"],
        "model.norm",
    )

    # ========================================================
    # LM head
    # ========================================================

    if not config.tie_word_embeddings:

        lm_head = params["lm_head"]

        _export_dense(
            tensors,
            lm_head,
            "lm_head",
        )

    return tensors
