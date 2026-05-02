"""
scripts/generate.py

Autoregressive text generation with LaughLM.

Supports:
  • KV cache for efficient O(1) per-token decoding
  • Top-k + top-p (nucleus) + temperature sampling
  • Greedy decoding (temperature=0)
  • Load from exported model (params.msgpack) or raw checkpoint

Usage:
    # From exported model:
    python -m scripts.generate \
        --model_dir exported_model \
        --prompt "The meaning of life is" \
        --max_tokens 100 \
        --temperature 0.8 \
        --top_k 50

    # From raw checkpoint:
    python -m scripts.generate \
        --checkpoint_dir checkpoints \
        --config configs/tpu_v5e_8.yaml \
        --prompt "Once upon a time" \
        --max_tokens 200

    # Interactive mode (no --prompt):
    python -m scripts.generate --model_dir exported_model
"""

import argparse
import json
from pathlib import Path
from typing import Optional, List

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_bytes

from LaughLM.config.loader import load_config
from LaughLM.config.schema import LaughLMConfig
from LaughLM.model.gpt import GPTModel
from LaughLM.model.layers.attention import KVCache, init_kv_cache


# ────────────────────────────────────────────────────────────────
# Sampling functions
# ────────────────────────────────────────────────────────────────

def sample_token(
    logits: jnp.ndarray,
    rng_key: jnp.ndarray,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> jnp.ndarray:
    """
    Sample a single token from logits with temperature, top-k, and top-p.

    Parameters
    ----------
    logits      : (V,) — raw logits for one position
    rng_key     : JAX PRNG key
    temperature : 0 = greedy, >0 = sampling
    top_k       : keep only top-k logits (0 = disabled)
    top_p       : nucleus sampling threshold (1.0 = disabled)

    Returns
    -------
    token : scalar int32
    """
    if temperature == 0.0:
        # Greedy
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    # Apply temperature
    logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < logits.shape[-1]:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
        # Set everything outside top-k to -inf
        logits = jnp.full_like(logits, -1e10)
        logits = logits.at[top_k_indices].set(top_k_logits)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_indices = jnp.argsort(logits, axis=-1)[::-1]
        sorted_logits = logits[sorted_indices]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs - jax.nn.softmax(sorted_logits, axis=-1) >= top_p
        sorted_logits = jnp.where(sorted_mask, -1e10, sorted_logits)

        # Unsort
        logits = jnp.empty_like(logits)
        logits = logits.at[sorted_indices].set(sorted_logits)

    # Sample from distribution
    token = jax.random.categorical(rng_key, logits, axis=-1)
    return token.astype(jnp.int32)


# ────────────────────────────────────────────────────────────────
# Generation loop
# ────────────────────────────────────────────────────────────────

def generate(
    model,
    params,
    input_ids: jnp.ndarray,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    seed: int = 42,
) -> List[int]:
    """
    Autoregressive generation with KV cache.

    Parameters
    ----------
    model          : GPTModel instance
    params         : model parameters
    input_ids      : (1, T) prompt token IDs
    max_new_tokens : max tokens to generate
    temperature    : sampling temperature
    top_k          : top-k filtering
    top_p          : nucleus sampling
    eos_token_id   : stop if this token is generated
    seed           : random seed

    Returns
    -------
    list of generated token IDs (including prompt)
    """
    config = model.config
    num_layers = config.model.num_layers
    num_kv_heads = config.model.num_kv_heads or config.model.num_heads
    head_dim = config.model.d_model // config.model.num_heads
    max_seq_len = config.model.max_seq_len

    rng = jax.random.PRNGKey(seed)

    # Collect all tokens
    generated = list(np.array(input_ids[0]))

    # ── Prefill: process entire prompt at once ────────────────
    # No KV cache for prefill — process full sequence
    logits, _ = model.apply({"params": params}, input_ids)

    # Get logits for last position
    next_logits = logits[0, -1, :]  # (V,)

    # ── Autoregressive decode: one token at a time ────────────
    # Initialize KV caches for all layers
    kv_caches = [
        init_kv_cache(1, max_seq_len, num_kv_heads, head_dim, jnp.bfloat16)
        for _ in range(num_layers)
    ]

    # Fill caches with prefill K/V by running prefill again with caches
    _, kv_caches = model.apply({"params": params}, input_ids, kv_caches=kv_caches)

    for step in range(max_new_tokens):
        # Sample next token
        rng, sample_key = jax.random.split(rng)
        next_token = sample_token(
            next_logits,
            sample_key,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        next_token_int = int(next_token)
        generated.append(next_token_int)

        # Check EOS
        if eos_token_id is not None and next_token_int == eos_token_id:
            break

        # Check max length
        if len(generated) >= max_seq_len:
            break

        # Feed single token through model with KV cache
        single_token = jnp.array([[next_token_int]], dtype=jnp.int32)
        logits, kv_caches = model.apply(
            {"params": params},
            single_token,
            kv_caches=kv_caches,
        )

        next_logits = logits[0, -1, :]

    return generated


# ────────────────────────────────────────────────────────────────
# Model loading
# ────────────────────────────────────────────────────────────────

def load_model_from_export(model_dir: str):
    """Load model from exported params.msgpack + config.json."""
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)
    config = LaughLMConfig(**config_dict)

    # Load params
    params_path = model_dir / "params.msgpack"
    if not params_path.exists():
        raise FileNotFoundError(f"Params not found: {params_path}")

    print(f"[generate] Loading params from {params_path}...")
    model = GPTModel(config=config)

    # Initialize to get structure
    dummy = jnp.zeros((1, 2), dtype=jnp.int32)
    init_params = model.init(jax.random.PRNGKey(0), dummy)["params"]

    # Load saved params
    with open(params_path, "rb") as f:
        params = from_bytes(init_params, f.read())

    print(f"[generate] Model loaded ({sum(x.size for x in jax.tree_util.tree_leaves(params)):,} params)")

    return model, params, config


def load_model_from_checkpoint(checkpoint_dir: str, config_path: str):
    """Load model from raw Orbax checkpoint."""
    from LaughLM.training.checkpoint import CheckpointManager
    from LaughLM.training.train_state import TrainState
    from LaughLM.training.optimizer import build_optimizer
    from LaughLM.training.scheduler import build_scheduler
    from LaughLM.utils.rng import create_rng

    config = load_config(config_path)
    model = GPTModel(config=config)
    rng = create_rng(seed=0)

    dummy = jnp.zeros((1, config.runtime.seq_len), dtype=jnp.int32)
    params = model.init(rng.next_key(), dummy)["params"]

    schedule = build_scheduler(config)
    optimizer = build_optimizer(config, schedule)
    opt_state = optimizer.init(params)

    target_state = TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        tokens_processed=0,
        rng_key=rng.key,
    )

    ckpt_manager = CheckpointManager(checkpoint_dir, max_to_keep=99)
    result = ckpt_manager.restore_latest(target_state=target_state)

    if result is None:
        raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")

    state, step = result
    print(f"[generate] Loaded checkpoint step {step} ({state.tokens_processed:,} tokens)")

    return model, state.params, config


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate text with LaughLM")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with exported model (params.msgpack + config.json)")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory with Orbax checkpoint (alternative to --model_dir)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (required with --checkpoint_dir)")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                        help="HuggingFace tokenizer name or path")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt (interactive mode if not provided)")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────
    if args.model_dir:
        model, params, config = load_model_from_export(args.model_dir)
    elif args.checkpoint_dir:
        if not args.config:
            raise ValueError("--config is required when using --checkpoint_dir")
        model, params, config = load_model_from_checkpoint(args.checkpoint_dir, args.config)
    else:
        raise ValueError("Provide either --model_dir or --checkpoint_dir")

    # ── Load tokenizer ────────────────────────────────────────
    from tokenizers import Tokenizer

    tokenizer_path = args.tokenizer
    if Path(tokenizer_path).exists():
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        # Try loading from HuggingFace
        from transformers import AutoTokenizer
        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer = hf_tokenizer

    def encode(text):
        if hasattr(tokenizer, 'encode'):
            result = tokenizer.encode(text)
            if hasattr(result, 'ids'):
                return result.ids
            return result
        return tokenizer(text)["input_ids"]

    def decode(ids):
        if hasattr(tokenizer, 'decode'):
            return tokenizer.decode(ids)
        return tokenizer.decode(ids)

    eos_id = None
    if hasattr(tokenizer, 'eos_token_id'):
        eos_id = tokenizer.eos_token_id
    elif hasattr(tokenizer, 'token_to_id'):
        eos_id = tokenizer.token_to_id("<|endoftext|>")

    # ── Generate ──────────────────────────────────────────────
    def run_generation(prompt_text: str):
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt_text}")
        print(f"{'─' * 60}")

        # Encode
        token_ids = encode(prompt_text)
        input_ids = jnp.array([token_ids], dtype=jnp.int32)

        print(f"[generate] Prompt tokens: {len(token_ids)}")
        print(f"[generate] Generating up to {args.max_tokens} tokens...")
        print(f"[generate] temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
        print()

        # Generate
        output_ids = generate(
            model=model,
            params=params,
            input_ids=input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            eos_token_id=eos_id,
            seed=args.seed,
        )

        # Decode
        output_text = decode(output_ids)
        new_text = decode(output_ids[len(token_ids):])

        print(f"{'═' * 60}")
        print(output_text)
        print(f"{'═' * 60}")
        print(f"\n[generate] Generated {len(output_ids) - len(token_ids)} new tokens")

    if args.prompt:
        run_generation(args.prompt)
    else:
        # Interactive mode
        print("\n🎭 LaughLM Interactive Generation")
        print("Type a prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if prompt.strip().lower() in ("quit", "exit", "q"):
                    break
                if prompt.strip():
                    run_generation(prompt)
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break


if __name__ == "__main__":
    main()
