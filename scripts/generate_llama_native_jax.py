from __future__ import annotations

import argparse
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from transformers import AutoTokenizer

from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.train_state import TrainState
from LaughLM.export.validate_hf import unbox_logically_partitioned


# ============================================================
# Sampling helpers
# ============================================================

def apply_repetition_penalty(
    logits,
    generated_tokens,
    repetition_penalty: float,
):
    if repetition_penalty is None or repetition_penalty <= 1.0:
        return logits

    for token_id in set(generated_tokens):
        token_id = int(token_id)

        token_logit = logits[token_id]

        # HF-style-ish repetition penalty:
        # positive logits divided, negative logits multiplied.
        new_logit = jnp.where(
            token_logit > 0,
            token_logit / repetition_penalty,
            token_logit * repetition_penalty,
        )

        logits = logits.at[token_id].set(new_logit)

    return logits


def apply_top_k(
    logits,
    top_k: int | None,
):
    if top_k is None or top_k <= 0:
        return logits

    top_k = min(top_k, logits.shape[-1])

    kth_value = jnp.sort(logits)[-top_k]

    logits = jnp.where(
        logits < kth_value,
        -jnp.inf,
        logits,
    )

    return logits


def apply_top_p(
    logits,
    top_p: float | None,
):
    if top_p is None or top_p >= 1.0:
        return logits

    sorted_idx = jnp.argsort(logits)[::-1]
    sorted_logits = logits[sorted_idx]

    sorted_probs = jax.nn.softmax(sorted_logits)
    cumulative_probs = jnp.cumsum(sorted_probs)

    remove_mask = cumulative_probs > top_p

    # Always keep the best token.
    remove_mask = remove_mask.at[0].set(False)

    sorted_logits = jnp.where(
        remove_mask,
        -jnp.inf,
        sorted_logits,
    )

    filtered_logits = (
        jnp.full_like(logits, -jnp.inf)
        .at[sorted_idx]
        .set(sorted_logits)
    )

    return filtered_logits


def sample_next_token(
    logits,
    rng,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
):
    if temperature is None or temperature <= 0.0:
        next_token = jnp.argmax(logits)
        return next_token, rng

    logits = logits / temperature

    logits = apply_top_k(
        logits,
        top_k,
    )

    logits = apply_top_p(
        logits,
        top_p,
    )

    rng, subkey = jax.random.split(rng)

    next_token = jax.random.categorical(
        subkey,
        logits,
    )

    return next_token, rng


# ============================================================
# Main native generator
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/v5e_pmap.yaml",
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    parser.add_argument(
        "--tokenizer",
        default="LaughTaleAI/LaughLM-v0.1",
        help="HF tokenizer repo/path. Can also use microsoft/Phi-3.5-mini-instruct.",
    )

    parser.add_argument(
        "--prompt",
        default="Once upon a time",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding. Overrides sampling temperature/top-k/top-p.",
    )

    args = parser.parse_args()

    # ========================================================
    # JAX devices
    # ========================================================

    print("\n================ JAX DEVICES ================\n")
    print(jax.devices())

    # ========================================================
    # Config
    # ========================================================

    print("\n================ LOAD CONFIG ================\n")

    exp_config = load_config(
        args.config,
    )

    llama_config = build_llama_config(
        exp_config,
    )

    seq_len = int(llama_config.max_position_embeddings)

    print("hidden_size:", llama_config.hidden_size)
    print("layers:", llama_config.num_hidden_layers)
    print("heads:", llama_config.num_attention_heads)
    print("kv_heads:", llama_config.num_key_value_heads)
    print("vocab:", llama_config.vocab_size)
    print("seq_len:", seq_len)
    print("bos:", llama_config.bos_token_id)
    print("eos:", llama_config.eos_token_id)
    print("pad:", llama_config.pad_token_id)

    # ========================================================
    # Tokenizer
    # ========================================================

    print("\n================ LOAD TOKENIZER ================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("tokenizer vocab_size:", tokenizer.vocab_size)
    print("tokenizer len:", len(tokenizer))
    print("tokenizer bos:", tokenizer.bos_token_id)
    print("tokenizer eos:", tokenizer.eos_token_id)
    print("tokenizer pad:", tokenizer.pad_token_id)

    if len(tokenizer) > llama_config.vocab_size:
        raise ValueError(
            f"Tokenizer len {len(tokenizer)} > model vocab {llama_config.vocab_size}"
        )

    eos_token_id = 32000
    bos_token_id = 1

    # ========================================================
    # Restore checkpoint
    # ========================================================

    print("\n================ RESTORE CHECKPOINT ================\n")

    checkpoints = CheckpointManager(
        args.checkpoint_dir,
    )

    restored = checkpoints.restore_latest(
        target_state=None,
    )

    if restored is None:
        raise RuntimeError(
            f"No checkpoint found in {args.checkpoint_dir}"
        )

    state, step = restored

    print(f"restored step: {step:,}")

    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state["params"]

    params = unbox_logically_partitioned(
        params,
    )

    print("params restored and unboxed")

    # ========================================================
    # Init native model
    # ========================================================

    print("\n================ INIT NATIVE LLAMA MODEL ================\n")

    model = LlamaForCausalLM(
        config=llama_config,
    )

    # ========================================================
    # JIT forward
    # ========================================================

    @jax.jit
    def forward(params, input_ids):
        logits, _ = model.apply(
            {"params": params},
            input_ids=input_ids,
            use_cache=False,
            mode="train",
        )

        return logits

    dummy = jnp.zeros(
        (1, seq_len),
        dtype=jnp.int32,
    )

    print("compiling forward...")
    _ = forward(params, dummy).block_until_ready()
    print("compiled")

    # ========================================================
    # Generate
    # ========================================================

    def generate(prompt: str):
        rng = jax.random.PRNGKey(
            args.seed,
        )

        prompt_tokens = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )

        if len(prompt_tokens) == 0:
            prompt_tokens = [bos_token_id]

        if len(prompt_tokens) >= seq_len:
            prompt_tokens = prompt_tokens[: seq_len - 1]

        generated_tokens = list(prompt_tokens)

        x = jnp.zeros(
            (1, seq_len),
            dtype=jnp.int32,
        )

        prompt_len = len(prompt_tokens)

        x = x.at[0, :prompt_len].set(
            jnp.asarray(
                prompt_tokens,
                dtype=jnp.int32,
            )
        )

        cur_len = prompt_len

        print("\n================ GENERATION ================\n")
        print(prompt, end="", flush=True)

        for _ in range(args.max_new_tokens):
            if cur_len >= seq_len:
                break

            logits = forward(
                params,
                x,
            )

            # Position cur_len - 1 predicts next token.
            next_logits = logits[0, cur_len - 1, :]

            next_logits = next_logits.astype(
                jnp.float32,
            )

            next_logits = apply_repetition_penalty(
                next_logits,
                generated_tokens,
                args.repetition_penalty,
            )

            if args.greedy:
                next_token = jnp.argmax(
                    next_logits,
                )
            else:
                next_token, rng = sample_next_token(
                    logits=next_logits,
                    rng=rng,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )

            next_token = int(
                jax.device_get(next_token)
            )

            if next_token == eos_token_id:
                print("\n\n[EOS]")
                break

            generated_tokens.append(
                next_token,
            )

            x = x.at[0, cur_len].set(
                next_token,
            )

            cur_len += 1

            piece = tokenizer.decode(
                [next_token],
                skip_special_tokens=True,
            )

            print(piece, end="", flush=True)

        text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return text

    final_text = generate(
        args.prompt,
    )

    print("\n\n================ FULL TEXT ================\n")
    print(final_text)


if __name__ == "__main__":
    main()
