"""
LaughLM/export/validate_hf.py

End-to-end Hugging Face export validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import jax
import jax.numpy as jnp

from flax import linen as nn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.llama.model import LlamaForCausalLM


# ============================================================
# Helpers
# ============================================================

def _to_numpy(x):
    return np.asarray(jax.device_get(x))


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def unbox_logically_partitioned(tree):
    """
    Convert Flax logical-partition wrappers / Orbax-restored boxed
    params into raw JAX arrays.

    Handles:
    - flax.linen.LogicallyPartitioned
    - Orbax-restored dicts like {"value": array, ...}
    """

    def is_boxed_value(x):
        return isinstance(x, dict) and "value" in x

    def is_leaf(x):
        return isinstance(x, nn.LogicallyPartitioned) or is_boxed_value(x)

    def unbox(x):
        if isinstance(x, nn.LogicallyPartitioned):
            return x.unbox(apply_constraint=False)

        if is_boxed_value(x):
            return x["value"]

        return x

    return jax.tree_util.tree_map(
        unbox,
        tree,
        is_leaf=is_leaf,
    )


# ============================================================
# Structural validation
# ============================================================

def validate_export_structure(hf_dir, config):
    hf_dir = Path(hf_dir)

    required_files = [
        "model.safetensors",
        "config.json",
        "generation_config.json",
    ]

    for filename in required_files:
        path = hf_dir / filename
        _assert(path.exists(), f"Missing required file: {filename}")

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    tokenizer_found = any((hf_dir / f).exists() for f in tokenizer_files)

    _assert(tokenizer_found, "Tokenizer files missing.")

    print("[validate] structure OK")


# ============================================================
# HF runtime validation
# ============================================================

def validate_hf_load(hf_dir):
    print("[validate] loading HF model...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)

    print("[validate] HF load OK")

    return model, tokenizer


# ============================================================
# Weight tying validation
# ============================================================

def validate_weight_tying(hf_model, config):
    if not config.tie_word_embeddings:
        print("[validate] untied embeddings")
        return

    input_embedding = hf_model.model.embed_tokens.weight
    output_embedding = hf_model.lm_head.weight

    same_ptr = input_embedding.data_ptr() == output_embedding.data_ptr()

    _assert(same_ptr, "HF tied embeddings broken.")

    print("[validate] tied embeddings OK")


# ============================================================
# Native LaughLM forward
# ============================================================

def run_native_forward(params, llama_config, input_ids):
    model = LlamaForCausalLM(config=llama_config)

    params = unbox_logically_partitioned(params)

    logits, _ = model.apply(
        {"params": params},
        input_ids=input_ids,
        use_cache=False,
        mode="train",
    )

    return _to_numpy(logits)


# ============================================================
# HF forward
# ============================================================

def run_hf_forward(hf_model, input_ids):
    torch_input_ids = torch.tensor(
        np.asarray(input_ids),
        dtype=torch.long,
    )

    with torch.no_grad():
        outputs = hf_model(input_ids=torch_input_ids)

    return outputs.logits.detach().cpu().float().numpy()


# ============================================================
# Logits comparison
# ============================================================

def compare_logits(native_logits, hf_logits, atol=2e-2):
    _assert(
        native_logits.shape == hf_logits.shape,
        (
            "Shape mismatch:\n"
            f"native={native_logits.shape}\n"
            f"hf={hf_logits.shape}"
        ),
    )

    diff = np.abs(native_logits - hf_logits)

    max_error = float(diff.max())
    mean_error = float(diff.mean())

    print("[validate] logits comparison")
    print(f"  max error : {max_error:.8f}")
    print(f"  mean error: {mean_error:.8f}")
    print(f"  atol      : {atol:.8f}")

    _assert(
        max_error < atol,
        (
            "HF parity failed.\n"
            f"max_error={max_error}\n"
            f"atol={atol}"
        ),
    )

    print("[validate] logits parity OK")

# ============================================================
# Generation smoke test
# ============================================================

def validate_generation(hf_model, tokenizer):
    prompt = "Hello"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
        )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print("[validate] generation output:")
    print(text)

    _assert(len(text) > 0, "Generation failed.")

    print("[validate] generation OK")


# ============================================================
# Full validation pipeline
# ============================================================

def validate_hf_export(
    *,
    hf_dir,
    config_path,
    params,
):
    exp_config = load_config(config_path)
    llama_config = build_llama_config(exp_config)

    params = unbox_logically_partitioned(params)

    validate_export_structure(hf_dir, llama_config)

    hf_model, tokenizer = validate_hf_load(hf_dir)

    validate_weight_tying(hf_model, llama_config)

    batch_size = 2
    seq_len = 16

    rng = np.random.default_rng(0)

    input_ids = rng.integers(
        low=0,
        high=llama_config.vocab_size,
        size=(batch_size, seq_len),
        dtype=np.int32,
    )

    input_ids_jax = jnp.asarray(input_ids)

    print("[validate] native forward...")

    native_logits = run_native_forward(
        params=params,
        llama_config=llama_config,
        input_ids=input_ids_jax,
    )

    print("[validate] HF forward...")

    hf_logits = run_hf_forward(
        hf_model,
        input_ids,
    )

    compare_logits(
        native_logits,
        hf_logits,
    )

    validate_generation(
        hf_model,
        tokenizer,
    )

    print("\n[validate] ALL CHECKS PASSED")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    from LaughLM.training.checkpoint import CheckpointManager
    from LaughLM.training.train_state import TrainState

    parser = argparse.ArgumentParser()

    parser.add_argument("--hf_dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint_dir", required=True)

    args = parser.parse_args()

    checkpoints = CheckpointManager(args.checkpoint_dir)

    restored = checkpoints.restore_latest(target_state=None)

    if restored is None:
        raise RuntimeError("No checkpoint found.")

    state, step = restored

    print(f"[validate] restored step={step:,}")

    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state["params"]

    params = unbox_logically_partitioned(params)

    validate_hf_export(
        hf_dir=args.hf_dir,
        config_path=args.config,
        params=params,
    )