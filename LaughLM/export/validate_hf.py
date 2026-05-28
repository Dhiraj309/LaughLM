"""
LaughLM/export/validate_hf.py

End-to-end Hugging Face export validation.

This validator is for EXPORT PARITY, not training performance.

Important:
- Production training may use bf16 compute and TPU Splash attention.
- HF validation should not compare TPU Splash directly against PyTorch attention.
- HF validation should avoid bf16/fp16 numerical noise while debugging mapping.
- Therefore this file forces:
    native attention_impl = "xla"
    native compute_dtype  = float32
    native output_dtype   = float32
    HF load dtype         = float32

After mapping parity is fixed, downstream generation/eval may still load HF in
float16/bfloat16 as desired.
"""

from __future__ import annotations

from dataclasses import replace
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


def _make_validation_config(llama_config):
    """
    Create a native config for export parity.

    Keep architecture/token IDs identical, but force numerically comparable
    validation settings.
    """

    return replace(
        llama_config,
        attention_impl="xla",
        attention_fallback="warn",
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


def unbox_logically_partitioned(tree):
    """
    Convert Flax logical-partition wrappers / Orbax-restored boxed params
    into raw JAX arrays.

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

        _assert(
            path.exists(),
            f"Missing required file: {filename}",
        )

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    tokenizer_found = any(
        (hf_dir / filename).exists()
        for filename in tokenizer_files
    )

    _assert(
        tokenizer_found,
        "Tokenizer files missing.",
    )

    print("[validate] structure OK")


# ============================================================
# HF runtime validation
# ============================================================


def validate_hf_load(hf_dir):
    print("[validate] loading HF model in float32...")

    # transformers 4.57 warns that torch_dtype is deprecated, but it still
    # works. Use dtype if available; fall back to torch_dtype for older builds.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        hf_dir
    )

    model.eval()

    print("[validate] HF load OK")
    print(f"[validate] HF parameter dtype: {next(model.parameters()).dtype}")

    return model, tokenizer


# ============================================================
# HF config validation
# ============================================================


def validate_hf_config_values(hf_model, config):
    hf_config = hf_model.config

    checks = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
    }

    for name, expected in checks.items():
        actual = getattr(
            hf_config,
            name,
            None,
        )

        _assert(
            actual == expected,
            (
                f"HF config mismatch for {name}:\n"
                f"  actual={actual!r}\n"
                f"  expected={expected!r}"
            ),
        )

    print("[validate] HF config values OK")


# ============================================================
# Weight tying validation
# ============================================================


def validate_weight_tying(hf_model, config):
    if not config.tie_word_embeddings:
        print("[validate] untied embeddings")
        return

    input_embedding = hf_model.model.embed_tokens.weight
    output_embedding = hf_model.lm_head.weight

    same_ptr = (
        input_embedding.data_ptr()
        == output_embedding.data_ptr()
    )

    _assert(
        same_ptr,
        "HF tied embeddings broken.",
    )

    print("[validate] tied embeddings OK")


# ============================================================
# Native LaughLM forward
# ============================================================


def run_native_forward(
    params,
    llama_config,
    input_ids,
):
    model = LlamaForCausalLM(
        config=llama_config
    )

    params = unbox_logically_partitioned(
        params
    )

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


def run_hf_forward(
    hf_model,
    input_ids,
):
    device = next(
        hf_model.parameters()
    ).device

    torch_input_ids = torch.tensor(
        np.asarray(input_ids),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = hf_model(
            input_ids=torch_input_ids,
            use_cache=False,
        )

    return (
        outputs
        .logits
        .detach()
        .cpu()
        .float()
        .numpy()
    )


# ============================================================
# Logit stats
# ============================================================


def compute_diff_stats(
    native_logits,
    hf_logits,
):
    _assert(
        native_logits.shape == hf_logits.shape,
        (
            "Shape mismatch:\n"
            f"native={native_logits.shape}\n"
            f"hf={hf_logits.shape}"
        ),
    )

    diff = np.abs(
        native_logits - hf_logits
    )

    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "p50": float(np.percentile(diff, 50.0)),
        "p95": float(np.percentile(diff, 95.0)),
        "p99": float(np.percentile(diff, 99.0)),
    }


def print_diff_stats(
    *,
    label,
    stats,
):
    print(f"[validate] {label} comparison")
    print(f"  max error : {stats['max']:.8f}")
    print(f"  mean error: {stats['mean']:.8f}")
    print(f"  p50 error : {stats['p50']:.8f}")
    print(f"  p95 error : {stats['p95']:.8f}")
    print(f"  p99 error : {stats['p99']:.8f}")


# ============================================================
# Diagnostic parity sweep
# ============================================================


def validate_logits_on_shape(
    *,
    params,
    llama_config,
    hf_model,
    batch_size,
    seq_len,
    vocab_size,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    input_ids = rng.integers(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=np.int32,
    )

    input_ids_jax = jnp.asarray(
        input_ids
    )

    print(
        "[validate] native forward "
        f"batch={batch_size} seq={seq_len}..."
    )

    native_logits = run_native_forward(
        params=params,
        llama_config=llama_config,
        input_ids=input_ids_jax,
    )

    print(
        "[validate] HF forward "
        f"batch={batch_size} seq={seq_len}..."
    )

    hf_logits = run_hf_forward(
        hf_model=hf_model,
        input_ids=input_ids,
    )

    stats = compute_diff_stats(
        native_logits,
        hf_logits,
    )

    label = (
        f"logits batch={batch_size} seq={seq_len}"
    )

    print_diff_stats(
        label=label,
        stats=stats,
    )

    return {
        "label": label,
        "stats": stats,
    }


def assert_diagnostic_results(results):
    """
    Interpret the diagnostic sweep.

    Expected after a correct export in fp32:
    - seq=1 should be extremely close.
    - longer seq should also be close.

    Failure interpretation:
    - seq=1 bad:
        dense/norm/mlp/embed/lm_head mapping or dtype issue.
    - seq=1 good but longer seq bad:
        attention/RoPE/mask/QK layout issue.
    """

    seq1 = results[0]["stats"]
    last = results[-1]["stats"]

    print("[validate] diagnostic summary:")

    for result in results:
        stats = result["stats"]
        print(
            f"  {result['label']}: "
            f"mean={stats['mean']:.8f}, "
            f"p99={stats['p99']:.8f}, "
            f"max={stats['max']:.8f}"
        )

    # fp32 parity thresholds.
    #
    # These are intentionally stricter than bf16 tolerance but not so strict
    # that one backend-level matmul rounding difference causes immediate noise.
    mean_tol = 2e-3
    p99_tol = 2e-2
    max_tol = 1e-1

    if (
        seq1["mean"] > mean_tol
        or seq1["p99"] > p99_tol
        or seq1["max"] > max_tol
    ):
        raise AssertionError(
            "HF parity failed already at seq=1.\n"
            "This is not a RoPE-over-time issue.\n"
            "Likely causes: Dense transpose, RMSNorm, MLP mapping, "
            "embedding/lm_head tying, or dtype mismatch.\n"
            f"seq1 stats={seq1}\n"
            f"thresholds: mean<={mean_tol}, "
            f"p99<={p99_tol}, max<={max_tol}"
        )

    if (
        last["mean"] > mean_tol
        or last["p99"] > p99_tol
        or last["max"] > max_tol
    ):
        raise AssertionError(
            "HF parity passed/mostly passed at seq=1 but failed at longer seq.\n"
            "Likely causes: RoPE convention, attention mask, or Q/K layout.\n"
            f"last stats={last}\n"
            f"thresholds: mean<={mean_tol}, "
            f"p99<={p99_tol}, max<={max_tol}"
        )

    print("[validate] logits parity OK")


# ============================================================
# Generation smoke test
# ============================================================


def validate_generation(
    hf_model,
    tokenizer,
):
    prompt = "Hello"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    device = next(
        hf_model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = hf_model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print("[validate] generation output:")
    print(text)

    _assert(
        len(text) > 0,
        "Generation failed.",
    )

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
    exp_config = load_config(
        config_path
    )

    train_llama_config = build_llama_config(
        exp_config
    )

    llama_config = _make_validation_config(
        train_llama_config
    )

    print("[validate] native validation config:")
    print(f"  attention_impl={llama_config.attention_impl}")
    print(f"  fused_qkv={llama_config.fused_qkv}")
    print(f"  tie_word_embeddings={llama_config.tie_word_embeddings}")
    print(f"  bos_token_id={llama_config.bos_token_id}")
    print(f"  eos_token_id={llama_config.eos_token_id}")
    print(f"  pad_token_id={llama_config.pad_token_id}")
    print(f"  param_dtype={llama_config.param_dtype}")
    print(f"  compute_dtype={llama_config.compute_dtype}")
    print(f"  output_dtype={llama_config.output_dtype}")

    params = unbox_logically_partitioned(
        params
    )

    validate_export_structure(
        hf_dir,
        llama_config,
    )

    hf_model, tokenizer = validate_hf_load(
        hf_dir
    )

    validate_hf_config_values(
        hf_model,
        train_llama_config,
    )

    validate_weight_tying(
        hf_model,
        train_llama_config,
    )

    results = []

    # seq=1:
    # Q/K/RoPE/causal masking mostly cannot explain a big error here.
    results.append(
        validate_logits_on_shape(
            params=params,
            llama_config=llama_config,
            hf_model=hf_model,
            batch_size=2,
            seq_len=1,
            vocab_size=llama_config.vocab_size,
            seed=0,
        )
    )

    # seq=16:
    # Enough to expose RoPE/QK/mask problems.
    results.append(
        validate_logits_on_shape(
            params=params,
            llama_config=llama_config,
            hf_model=hf_model,
            batch_size=2,
            seq_len=16,
            vocab_size=llama_config.vocab_size,
            seed=1,
        )
    )

    # seq=128:
    # Still cheap, closer to real attention behavior.
    results.append(
        validate_logits_on_shape(
            params=params,
            llama_config=llama_config,
            hf_model=hf_model,
            batch_size=1,
            seq_len=128,
            vocab_size=llama_config.vocab_size,
            seed=2,
        )
    )

    assert_diagnostic_results(
        results
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

    parser.add_argument(
        "--hf_dir",
        required=True,
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    args = parser.parse_args()

    checkpoints = CheckpointManager(
        args.checkpoint_dir
    )

    restored = checkpoints.restore_latest(
        target_state=None,
    )

    if restored is None:
        raise RuntimeError(
            "No checkpoint found."
        )

    state, step = restored

    print(f"[validate] restored step={step:,}")

    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state["params"]

    params = unbox_logically_partitioned(
        params
    )

    validate_hf_export(
        hf_dir=args.hf_dir,
        config_path=args.config,
        params=params,
    )
