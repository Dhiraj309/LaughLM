"""
scripts/benchmark.py

Frontier LLM Evaluation Suite for LaughLM (JAX-native).

Runs standard benchmarks directly on the Flax model without conversion:
  • WikiText-2 Perplexity (rolling, stride=512)
  • LAMBADA (OpenAI) — last-word prediction accuracy + perplexity
  • HellaSwag — 4-choice sentence completion (acc_norm)
  • PIQA — 2-choice physical intuition QA (acc)
  • ARC-Easy — 4-choice science QA (acc_norm)

All multiple-choice benchmarks use log-probability scoring:
  score(choice) = sum(log P(token_i | context + tokens_<i)) / len(tokens)

Reference baselines (zero-shot):
  ┌───────────────┬──────────┬────────────┬────────────┐
  │ Benchmark     │ Random   │ GPT-2 124M │ Pythia-160M│
  ├───────────────┼──────────┼────────────┼────────────┤
  │ WikiText-2    │   N/A    │   29.41    │    ~30     │
  │ LAMBADA acc   │   ~0%    │   45.99%   │   32.8%    │
  │ HellaSwag     │   25%    │   ~31%     │   30.2%    │
  │ PIQA          │   50%    │   ~63%     │   62.7%    │
  │ ARC-Easy      │   25%    │   ~43%     │   43.5%    │
  └───────────────┴──────────┴────────────┴────────────┘

Usage:
    python -m scripts.benchmark \
        --model_dir exported_model \
        --tokenizer gpt2 \
        --benchmarks wikitext2,lambada,hellaswag,piqa,arc_easy \
        --output_file results.json

    python -m scripts.benchmark \
        --checkpoint_dir checkpoints \
        --config configs/tpu_v5e_8.yaml \
        --benchmarks lambada,piqa
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax.serialization import from_bytes


# ────────────────────────────────────────────────────────────────
# Model loading (reused from generate.py)
# ────────────────────────────────────────────────────────────────

def load_model(args):
    """Load model from export dir or checkpoint."""
    from LaughLM.config.schema import LaughLMConfig
    from LaughLM.model.gpt import GPTModel

    if args.model_dir:
        model_dir = Path(args.model_dir)
        with open(model_dir / "config.json") as f:
            config = LaughLMConfig(**json.load(f))

        model = GPTModel(config=config)
        dummy = jnp.zeros((1, 2), dtype=jnp.int32)
        init_params = model.init(jax.random.PRNGKey(0), dummy)["params"]

        with open(model_dir / "params.msgpack", "rb") as f:
            params = from_bytes(init_params, f.read())

    elif args.checkpoint_dir:
        from LaughLM.config.loader import load_config
        from LaughLM.training.checkpoint import CheckpointManager
        from LaughLM.training.train_state import TrainState
        from LaughLM.training.optimizer import build_optimizer
        from LaughLM.training.scheduler import build_scheduler
        from LaughLM.utils.rng import create_rng

        config = load_config(args.config)
        model = GPTModel(config=config)
        rng = create_rng(seed=0)

        dummy = jnp.zeros((1, config.runtime.seq_len), dtype=jnp.int32)
        params = model.init(rng.next_key(), dummy)["params"]

        schedule = build_scheduler(config)
        optimizer = build_optimizer(config, schedule)
        opt_state = optimizer.init(params)

        target_state = TrainState(
            params=params, opt_state=opt_state,
            step=0, tokens_processed=0, rng_key=rng.key,
        )

        ckpt = CheckpointManager(args.checkpoint_dir, max_to_keep=99)
        result = ckpt.restore_latest(target_state=target_state)
        if result is None:
            raise RuntimeError(f"No checkpoint in {args.checkpoint_dir}")
        state, step = result
        params = state.params
        print(f"[benchmark] Loaded checkpoint step {step}")
    else:
        raise ValueError("Provide --model_dir or --checkpoint_dir")

    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[benchmark] Model loaded: {total_params:,} parameters")

    return model, params, config


# ────────────────────────────────────────────────────────────────
# Core scoring function
# ────────────────────────────────────────────────────────────────

def compute_logprobs(model, params, token_ids: List[int], max_len: int) -> np.ndarray:
    """
    Compute per-token log-probabilities for a sequence.

    Returns log P(token_i | tokens_<i) for i = 1..len-1.
    Shape: (len - 1,)
    """
    # Truncate to max model length
    token_ids = token_ids[:max_len]
    if len(token_ids) < 2:
        return np.array([])

    input_ids = jnp.array([token_ids], dtype=jnp.int32)
    logits, _ = model.apply({"params": params}, input_ids)

    # logits[0, i, :] predicts token at position i+1
    log_probs = jax.nn.log_softmax(logits[0], axis=-1)

    # Gather log-prob of actual next tokens
    targets = np.array(token_ids[1:])
    positions = np.arange(len(targets))

    token_logprobs = np.array(log_probs[positions, targets])
    return token_logprobs


# ────────────────────────────────────────────────────────────────
# WikiText-2 Perplexity
# ────────────────────────────────────────────────────────────────

def eval_wikitext2(model, params, tokenizer, max_len: int, stride: int = 512) -> Dict:
    """Compute perplexity on WikiText-2 test set using sliding window."""
    from datasets import load_dataset

    print("\n[wikitext2] Loading dataset...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer.encode(text)
    print(f"[wikitext2] Total tokens: {len(tokens):,}")

    total_nll = 0.0
    total_tokens = 0

    # Sliding window
    for start in range(0, len(tokens) - 1, stride):
        end = min(start + max_len, len(tokens))
        chunk = tokens[start:end]

        if len(chunk) < 2:
            continue

        input_ids = jnp.array([chunk], dtype=jnp.int32)
        logits, _ = model.apply({"params": params}, input_ids)
        log_probs = jax.nn.log_softmax(logits[0], axis=-1)

        # Only score tokens after the overlap (except first window)
        score_start = 0 if start == 0 else (max_len - stride - 1)
        targets = np.array(chunk[score_start + 1:])
        positions = np.arange(score_start, score_start + len(targets))

        if len(targets) == 0:
            continue

        nll = -float(jnp.sum(log_probs[positions, targets]))
        total_nll += nll
        total_tokens += len(targets)

    ppl = float(np.exp(total_nll / max(total_tokens, 1)))
    print(f"[wikitext2] PPL: {ppl:.2f} (tokens scored: {total_tokens:,})")

    return {"perplexity": ppl, "tokens_scored": total_tokens}


# ────────────────────────────────────────────────────────────────
# LAMBADA (OpenAI)
# ────────────────────────────────────────────────────────────────

def eval_lambada(model, params, tokenizer, max_len: int) -> Dict:
    """LAMBADA: predict the last word given context."""
    from datasets import load_dataset

    print("\n[lambada] Loading dataset...")
    ds = load_dataset("EleutherAI/lambada_openai", "default", split="test")

    correct = 0
    total = 0
    total_nll = 0.0
    total_target_tokens = 0

    for sample in ds:
        text = sample["text"]
        tokens = tokenizer.encode(text)

        if len(tokens) < 2 or len(tokens) > max_len:
            continue

        # Last word = last token(s) after the last space
        # LAMBADA convention: score last word as continuation
        words = text.rsplit(" ", 1)
        if len(words) < 2:
            continue

        context = words[0] + " "
        target_word = words[1]

        ctx_tokens = tokenizer.encode(context)
        full_tokens = tokenizer.encode(text)
        target_tokens = full_tokens[len(ctx_tokens):]

        if len(target_tokens) == 0:
            continue

        # Compute log-probs for full sequence
        input_ids = jnp.array([full_tokens], dtype=jnp.int32)
        logits, _ = model.apply({"params": params}, input_ids)
        log_probs = jax.nn.log_softmax(logits[0], axis=-1)

        # Check if model predicts target tokens greedily
        pred_correct = True
        nll = 0.0
        for i, t in enumerate(target_tokens):
            pos = len(ctx_tokens) - 1 + i
            if pos >= len(full_tokens) - 1:
                break
            pred = int(jnp.argmax(log_probs[pos]))
            if pred != t:
                pred_correct = False
            nll -= float(log_probs[pos, t])

        if pred_correct:
            correct += 1

        total += 1
        total_nll += nll
        total_target_tokens += len(target_tokens)

    acc = correct / max(total, 1) * 100
    ppl = float(np.exp(total_nll / max(total_target_tokens, 1)))
    print(f"[lambada] Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"[lambada] Perplexity: {ppl:.2f}")

    return {"accuracy": acc, "perplexity": ppl, "total": total, "correct": correct}


# ────────────────────────────────────────────────────────────────
# Multiple-choice benchmark (generic)
# ────────────────────────────────────────────────────────────────

def eval_multiple_choice(
    model, params, tokenizer, max_len: int,
    dataset_name: str, dataset_config: str, split: str,
    context_fn, choices_fn, label_fn,
    task_name: str, normalize: bool = True,
) -> Dict:
    """
    Generic multiple-choice evaluation via log-prob scoring.

    For each sample:
      1. Build context string
      2. For each choice, compute: score = sum(log P(choice_token_i | context + choice_<i))
      3. If normalize=True: score /= len(choice_tokens)  (acc_norm)
      4. Prediction = argmax(scores)
    """
    from datasets import load_dataset

    print(f"\n[{task_name}] Loading dataset...")
    ds = load_dataset(dataset_name, dataset_config, split=split)

    correct = 0
    total = 0

    for sample in ds:
        context = context_fn(sample)
        choices = choices_fn(sample)
        label = label_fn(sample)

        if not choices or label is None:
            continue

        scores = []
        for choice in choices:
            full_text = context + choice
            full_tokens = tokenizer.encode(full_text)
            ctx_tokens = tokenizer.encode(context)
            choice_tokens = full_tokens[len(ctx_tokens):]

            if len(choice_tokens) == 0 or len(full_tokens) > max_len:
                scores.append(-1e10)
                continue

            # Score the choice continuation
            input_ids = jnp.array([full_tokens], dtype=jnp.int32)
            logits, _ = model.apply({"params": params}, input_ids)
            log_probs = jax.nn.log_softmax(logits[0], axis=-1)

            choice_logprob = 0.0
            for i, t in enumerate(choice_tokens):
                pos = len(ctx_tokens) - 1 + i
                if pos < logits.shape[1] - 1:
                    choice_logprob += float(log_probs[pos, t])

            if normalize and len(choice_tokens) > 0:
                choice_logprob /= len(choice_tokens)

            scores.append(choice_logprob)

        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
        total += 1

    acc = correct / max(total, 1) * 100
    print(f"[{task_name}] Accuracy: {acc:.2f}% ({correct}/{total})")

    return {"accuracy": acc, "total": total, "correct": correct, "normalized": normalize}


# ────────────────────────────────────────────────────────────────
# Benchmark wrappers
# ────────────────────────────────────────────────────────────────

def eval_hellaswag(model, params, tokenizer, max_len):
    """HellaSwag: 4-choice sentence completion (acc_norm)."""
    return eval_multiple_choice(
        model, params, tokenizer, max_len,
        dataset_name="Rowan/hellaswag", dataset_config="default", split="validation",
        context_fn=lambda s: s["ctx"],
        choices_fn=lambda s: s["endings"],
        label_fn=lambda s: int(s["label"]),
        task_name="hellaswag",
        normalize=True,
    )


def eval_piqa(model, params, tokenizer, max_len):
    """PIQA: 2-choice physical intuition QA."""
    return eval_multiple_choice(
        model, params, tokenizer, max_len,
        dataset_name="ybisk/piqa", dataset_config="plain_text", split="validation",
        context_fn=lambda s: "Question: " + s["goal"] + "\nAnswer:",
        choices_fn=lambda s: [" " + s["sol1"], " " + s["sol2"]],
        label_fn=lambda s: int(s["label"]),
        task_name="piqa",
        normalize=True,
    )


def eval_arc_easy(model, params, tokenizer, max_len):
    """ARC-Easy: 4-choice science QA (acc_norm)."""
    return eval_multiple_choice(
        model, params, tokenizer, max_len,
        dataset_name="allenai/ai2_arc", dataset_config="ARC-Easy", split="test",
        context_fn=lambda s: "Question: " + s["question"] + "\nAnswer:",
        choices_fn=lambda s: [" " + c for c in s["choices"]["text"]],
        label_fn=lambda s: s["choices"]["label"].index(s["answerKey"]) if s["answerKey"] in s["choices"]["label"] else None,
        task_name="arc_easy",
        normalize=True,
    )


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

BENCHMARKS = {
    "wikitext2": eval_wikitext2,
    "lambada": eval_lambada,
    "hellaswag": eval_hellaswag,
    "piqa": eval_piqa,
    "arc_easy": eval_arc_easy,
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark LaughLM")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--benchmarks", type=str, default="wikitext2,lambada,hellaswag,piqa,arc_easy",
                        help="Comma-separated benchmark names")
    parser.add_argument("--output_file", type=str, default="benchmark_results.json")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per benchmark (for testing)")
    args = parser.parse_args()

    # Load model
    model, params, config = load_model(args)
    max_len = config.model.max_seq_len

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"[benchmark] Tokenizer: {args.tokenizer} (vocab={tokenizer.vocab_size:,})")

    # Run benchmarks
    benchmark_list = [b.strip() for b in args.benchmarks.split(",")]
    results = {}

    print(f"\n{'═' * 60}")
    print(f" LaughLM Benchmark Suite")
    print(f" Model: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,} params")
    print(f" Benchmarks: {', '.join(benchmark_list)}")
    print(f"{'═' * 60}")

    start_time = time.time()

    for bench_name in benchmark_list:
        if bench_name not in BENCHMARKS:
            print(f"\n[WARNING] Unknown benchmark: {bench_name}. Skipping.")
            continue

        bench_fn = BENCHMARKS[bench_name]
        bench_start = time.time()

        if bench_name == "wikitext2":
            result = bench_fn(model, params, tokenizer, max_len)
        elif bench_name == "lambada":
            result = bench_fn(model, params, tokenizer, max_len)
        else:
            result = bench_fn(model, params, tokenizer, max_len)

        result["time_seconds"] = time.time() - bench_start
        results[bench_name] = result

    total_time = time.time() - start_time

    # ── Print results table ───────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f" RESULTS")
    print(f"{'═' * 60}")
    print(f"{'Benchmark':<15} {'Metric':<12} {'Score':>10} {'Baseline (GPT-2)':>18}")
    print(f"{'─' * 60}")

    baselines = {
        "wikitext2": ("PPL ↓", 29.41),
        "lambada": ("Acc ↑", 45.99),
        "hellaswag": ("Acc_norm ↑", 31.0),
        "piqa": ("Acc ↑", 63.0),
        "arc_easy": ("Acc_norm ↑", 43.0),
    }

    for bench_name, result in results.items():
        if bench_name == "wikitext2":
            score = f"{result['perplexity']:.2f}"
        else:
            score = f"{result['accuracy']:.2f}%"

        metric, baseline = baselines.get(bench_name, ("?", "?"))
        baseline_str = f"{baseline}" if isinstance(baseline, float) else str(baseline)
        if bench_name != "wikitext2":
            baseline_str += "%"

        print(f"{bench_name:<15} {metric:<12} {score:>10} {baseline_str:>18}")

    print(f"{'─' * 60}")
    print(f"Total eval time: {total_time:.1f}s")
    print(f"{'═' * 60}")

    # Save results
    output = {
        "model_params": sum(x.size for x in jax.tree_util.tree_leaves(params)),
        "benchmarks": results,
        "total_time_seconds": total_time,
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    main()