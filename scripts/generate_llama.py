"""
scripts/generate_llama.py

Canonical autoregressive generation runtime for LaughLM Llama models.

Design goals
------------
- HF-style prefill/decode lifecycle
- deterministic KV-cache semantics
- explicit decode positions
- minimal runtime abstraction surface
- future-compatible with:
    - FlashAttention
    - paged KV cache
    - TPU decode compilation
    - speculative decoding

Generation lifecycle
--------------------
1. Prefill full prompt
2. Build KV cache
3. Decode one token at a time

Tensor conventions
------------------
input_ids:
    [B, T]

positions:
    [B, T]

logits:
    [B, T, V]
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from flax.serialization import from_bytes

from LaughLM.model.llama.config import (
    LlamaConfig,
)

from LaughLM.model.llama.model import (
    LlamaForCausalLM,
)

from LaughLM.model.llama.kv_cache import (
    init_kv_cache,
)

from LaughLM.model.llama.sampling import (
    sample_next_token,
)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_model(
    model_dir: str,
):
    """
    Load exported Llama model.

    Expected files
    --------------
    config.json
    params.msgpack
    """

    model_dir = Path(model_dir)

    config_path = model_dir / "config.json"

    params_path = model_dir / "params.msgpack"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not params_path.exists():
        raise FileNotFoundError(params_path)

    with open(config_path) as f:
        config_dict = json.load(f)

    config = LlamaConfig(**config_dict)

    model = LlamaForCausalLM(config)

    dummy_input_ids = jnp.zeros(
        (1, 1),
        dtype=jnp.int32,
    )

    dummy_positions = jnp.zeros(
        (1, 1),
        dtype=jnp.int32,
    )

    variables = model.init(
        jax.random.PRNGKey(0),
        input_ids=dummy_input_ids,
        positions=dummy_positions,
        mode="train",
    )

    with open(params_path, "rb") as f:
        params = from_bytes(
            variables["params"],
            f.read(),
        )

    return model, params, config


# ─────────────────────────────────────────────
# KV cache initialization
# ─────────────────────────────────────────────

def create_kv_caches(
    config: LlamaConfig,
    batch_size: int,
):
    """
    Create per-layer static KV caches.
    """

    caches = []

    for _ in range(config.num_hidden_layers):

        cache = init_kv_cache(
            batch_size=batch_size,
            max_seq_len=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=jnp.bfloat16,
        )

        caches.append(cache)

    return caches


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate(
    model,
    params,
    input_ids: jnp.ndarray,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    seed: int = 0,
):
    """
    Generate autoregressively from prompt.

    Parameters
    ----------
    input_ids:
        [B, T_prompt]
    """

    batch_size, prompt_length = input_ids.shape

    if batch_size != 1:
        raise NotImplementedError(
            "Only batch_size=1 currently supported."
        )

    rng = jax.random.PRNGKey(seed)

    generated_ids = list(
        np.array(input_ids[0])
    )

    # ─────────────────────────────────────────
    # Initialize KV caches
    # ─────────────────────────────────────────

    kv_caches = create_kv_caches(
        model.config,
        batch_size=batch_size,
    )

    # ─────────────────────────────────────────
    # Prefill
    # ─────────────────────────────────────────

    positions = jnp.arange(
        prompt_length,
        dtype=jnp.int32,
    )[None, :]

    logits, kv_caches = model.apply(
        {"params": params},
        input_ids=input_ids,
        positions=positions,
        kv_caches=kv_caches,
        mode="prefill",
    )

    next_token_logits = logits[:, -1, :]

    # ─────────────────────────────────────────
    # Decode loop
    # ─────────────────────────────────────────

    for _ in range(max_new_tokens):

        rng, sample_rng = jax.random.split(rng)

        next_token = sample_next_token(
            logits=next_token_logits,
            rng=sample_rng,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        next_token_int = int(next_token[0])

        generated_ids.append(
            next_token_int
        )

        # ─────────────────────────────────────
        # EOS stopping
        # ─────────────────────────────────────

        if (
            eos_token_id is not None
            and next_token_int == eos_token_id
        ):
            break

        # ─────────────────────────────────────
        # Max length stopping
        # ─────────────────────────────────────

        current_length = (
            kv_caches[0].cache_position
        )

        if (
            current_length
            >= model.config.max_position_embeddings
        ):
            break

        # ─────────────────────────────────────
        # Decode one token
        # ─────────────────────────────────────

        decode_input_ids = jnp.array(
            [[next_token_int]],
            dtype=jnp.int32,
        )

        decode_positions = jnp.array(
            [[current_length]],
            dtype=jnp.int32,
        )

        logits, kv_caches = model.apply(
            {"params": params},
            input_ids=decode_input_ids,
            positions=decode_positions,
            kv_caches=kv_caches,
            mode="decode",
        )

        next_token_logits = logits[:, -1, :]

    return generated_ids


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # ─────────────────────────────────────────
    # Load tokenizer
    # ─────────────────────────────────────────

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
    )

    # ─────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────

    model, params, config = load_model(
        args.model_dir,
    )

    # ─────────────────────────────────────────
    # Encode prompt
    # ─────────────────────────────────────────

    input_ids = tokenizer.encode(
        args.prompt,
        return_tensors="np",
    )

    input_ids = jnp.array(
        input_ids,
        dtype=jnp.int32,
    )

    # ─────────────────────────────────────────
    # Generate
    # ─────────────────────────────────────────

    output_ids = generate(
        model=model,
        params=params,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=config.eos_token_id,
        seed=args.seed,
    )

    # ─────────────────────────────────────────
    # Decode output
    # ─────────────────────────────────────────

    text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
    )

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()