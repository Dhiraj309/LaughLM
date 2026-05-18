"""
scripts/generate_llama.py

Frontier-grade autoregressive generation runtime.

Frontier-grade additions
────────────────────────────────────────────
1. Explicit prefill/decode compilation
2. Stable KV-cache semantics
3. Decode-specialized jit graphs
4. Compile-safe decode positions
5. Deterministic sampling
6. Future-ready TPU decode runtime
"""

import argparse
import json

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from flax.serialization import (
    from_bytes,
)

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
# Load model
# ─────────────────────────────────────────────

def load_model(
    model_dir: str,
):

    model_dir = Path(model_dir)

    config_path = (
        model_dir / "config.json"
    )

    params_path = (
        model_dir / "params.msgpack"
    )

    with open(config_path) as f:
        config_dict = json.load(f)

    config = LlamaConfig(
        **config_dict
    )

    model = LlamaForCausalLM(
        config=config
    )

    dummy_input_ids = jnp.zeros(
        (1, 1),
        dtype=jnp.int32,
    )

    variables = model.init(
        jax.random.PRNGKey(0),
        input_ids=dummy_input_ids,
        use_cache=False,
        mode="train",
    )

    with open(params_path, "rb") as f:

        params = from_bytes(
            variables["params"],
            f.read(),
        )

    return (
        model,
        params,
        config,
    )


# ─────────────────────────────────────────────
# KV caches
# ─────────────────────────────────────────────

def create_kv_caches(
    config,
    batch_size,
):

    return [
        init_kv_cache(
            batch_size=batch_size,
            max_seq_len=(
                config.max_position_embeddings
            ),
            num_kv_heads=(
                config.num_key_value_heads
            ),
            head_dim=config.head_dim,
            dtype=config.compute_dtype,
        )
        for _ in range(
            config.num_hidden_layers
        )
    ]


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate(
    model,
    params,
    input_ids,
    max_new_tokens=128,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    eos_token_id=None,
    seed=0,
):

    batch_size, prompt_length = (
        input_ids.shape
    )

    if batch_size != 1:

        raise NotImplementedError(
            "Only batch_size=1 "
            "currently supported."
        )

    rng = jax.random.PRNGKey(
        seed
    )

    generated_ids = list(
        np.asarray(input_ids[0])
    )

    kv_caches = create_kv_caches(
        model.config,
        batch_size,
    )

    # ==================================================
    # Prefill function
    # ==================================================

    @jax.jit
    def prefill_step(
        params,
        input_ids,
        kv_caches,
    ):

        logits, kv_caches = (
            model.apply(
                {"params": params},
                input_ids=input_ids,
                kv_caches=kv_caches,
                use_cache=True,
                mode="prefill",
            )
        )

        return (
            logits,
            kv_caches,
        )

    # ==================================================
    # Decode function
    # ==================================================

    @jax.jit
    def decode_step(
        params,
        input_ids,
        kv_caches,
    ):

        logits, kv_caches = (
            model.apply(
                {"params": params},
                input_ids=input_ids,
                kv_caches=kv_caches,
                use_cache=True,
                mode="decode",
            )
        )

        return (
            logits,
            kv_caches,
        )

    # ==================================================
    # Prefill
    # ==================================================

    logits, kv_caches = (
        prefill_step(
            params,
            input_ids,
            kv_caches,
        )
    )

    next_token_logits = (
        logits[:, -1, :]
    )

    # ==================================================
    # Decode loop
    # ==================================================

    for _ in range(max_new_tokens):

        rng, sample_rng = (
            jax.random.split(rng)
        )

        next_token = (
            sample_next_token(
                logits=next_token_logits,
                rng=sample_rng,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )

        next_token_int = int(
            next_token[0]
        )

        generated_ids.append(
            next_token_int
        )

        # ------------------------------------------------
        # EOS
        # ------------------------------------------------

        if (
            eos_token_id is not None
            and next_token_int
            == eos_token_id
        ):
            break

        # ------------------------------------------------
        # Max length
        # ------------------------------------------------

        current_length = int(
            kv_caches[0]
            .cache_position
        )

        if (
            current_length
            >= model.config
            .max_position_embeddings
        ):
            break

        # ------------------------------------------------
        # Decode token
        # ------------------------------------------------

        decode_input_ids = jnp.asarray(
            [[next_token_int]],
            dtype=jnp.int32,
        )

        logits, kv_caches = (
            decode_step(
                params,
                decode_input_ids,
                kv_caches,
            )
        )

        next_token_logits = (
            logits[:, -1, :]
        )

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

    from transformers import (
        AutoTokenizer
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            args.tokenizer
        )
    )

    model, params, config = (
        load_model(
            args.model_dir
        )
    )

    input_ids = tokenizer.encode(
        args.prompt,
        return_tensors="np",
    )

    input_ids = jnp.asarray(
        input_ids,
        dtype=jnp.int32,
    )

    output_ids = generate(
        model=model,
        params=params,
        input_ids=input_ids,
        max_new_tokens=(
            args.max_new_tokens
        ),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=(
            config.eos_token_id
        ),
        seed=args.seed,
    )

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
