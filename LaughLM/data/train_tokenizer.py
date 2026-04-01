
import json
import os
from typing import Iterable, List, Optional
from multiprocessing import Process, Queue, cpu_count

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel, Digits, Punctuation, Sequence
from tokenizers.normalizers import (
    NFKC,
    Replace,
    Sequence as NormSequence,
    Strip,
)

# ── MAX CPU UTILIZATION ───────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_RS_NUM_CPUS"]      = str(cpu_count())

# ── Special tokens (order = ID assignment, do not reorder) ────
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
PAD_ID, EOS_ID, BOS_ID, UNK_ID = 0, 1, 2, 3

_MAX_WORD_LEN = 50


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def _validate_vocab_size(vocab_size: int):
    assert vocab_size % 128 == 0, (
        f"vocab_size={vocab_size} must be divisible by 128 (MXU alignment)"
    )


def _truncate_long_words(text: str, max_len: int = _MAX_WORD_LEN) -> str:
    """Truncate long tokens (URLs, LaTeX, base64) to prevent vocab pollution."""
    return " ".join(
        w if len(w) <= max_len else w[:max_len]
        for w in text.split()
    )


def _get_parquet_files(dataset_name: str) -> List[str]:
    """Get sorted list of all parquet filenames from an HF dataset repo."""
    api = HfApi()
    all_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted(f for f in all_files if f.endswith(".parquet"))
    print(f"  Found {len(parquet_files)} parquet files in {dataset_name}")
    return parquet_files


def _split_files(files: List[str], num_workers: int) -> List[List[str]]:
    """Split file list into num_workers roughly equal chunks."""
    chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        chunks[i % num_workers].append(f)
    return chunks


# ─────────────────────────────────────────────────────────────
# Worker — file-level sharding (no duplicate data)
# ─────────────────────────────────────────────────────────────

def _producer_worker(
    result_queue: Queue,
    dataset_name: str,
    file_list:    List[str],
    worker_id:    int,
    per_worker:   int,
    min_text_len: int,
    tmp_path:     str,
):
    """
    Worker process: reads only its assigned parquet files.

    File-level sharding guarantees:
    - Each row is read exactly once across all workers (no duplicates)
    - No row-skipping overhead
    - Total I/O = dataset size, not dataset_size × num_workers

    try/finally guarantees result_queue.put() always runs,
    even if the worker crashes — prevents main process deadlock.
    """
    written = 0
    skipped = 0

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for filename in file_list:

                if written >= per_worker:
                    break

                # Load one parquet file at a time — minimal RAM per worker
                ds = load_dataset(
                    dataset_name,
                    split      = "train",
                    data_files = filename,
                    streaming  = True,
                )

                for sample in ds:
                    if written >= per_worker:
                        break

                    text = sample.get("text", "")
                    if not text or len(text) < min_text_len:
                        skipped += 1
                        continue

                    text = _truncate_long_words(text)
                    f.write(text.replace("\n", " ").strip() + "\n")
                    written += 1

    except Exception as e:
        print(f"[worker {worker_id}] error: {e}", flush=True)

    finally:
        # Always signals done — prevents deadlock if worker crashes
        result_queue.put({
            "worker_id": worker_id,
            "tmp_path":  tmp_path,
            "written":   written,
            "skipped":   skipped,
        })


# ─────────────────────────────────────────────────────────────
# Stage 1 — Parallel file collection
# ─────────────────────────────────────────────────────────────

def _collect_to_files(
    dataset_name:  str,
    total_samples: int,
    num_workers:   int,
    min_text_len:  int,
    work_dir:      str,
) -> List[str]:
    """
    Split dataset parquet files across workers, collect to text files.
    Returns list of non-empty temp file paths.
    """
    # Cap workers to available CPUs
    actual_workers = min(num_workers, cpu_count())
    if actual_workers < num_workers:
        print(f"  ⚠ Capping workers: {num_workers} → {actual_workers} "
              f"(only {cpu_count()} CPUs available)")

    parquet_files = _get_parquet_files(dataset_name)
    file_chunks   = _split_files(parquet_files, actual_workers)
    per_worker    = (total_samples + actual_workers - 1) // actual_workers

    print(f"\n[Stage 1] Collecting {total_samples:,} samples")
    print(f"  Workers        : {actual_workers}")
    print(f"  Files/worker   : ~{len(parquet_files) // actual_workers}")
    print(f"  Samples/worker : {per_worker:,}")
    print(f"  Work dir       : {work_dir}")

    os.makedirs(work_dir, exist_ok=True)

    result_queue = Queue()
    processes    = []

    for worker_id in range(actual_workers):
        tmp_path = os.path.join(work_dir, f"worker_{worker_id:04d}.txt")
        p = Process(
            target = _producer_worker,
            args   = (
                result_queue,
                dataset_name,
                file_chunks[worker_id],
                worker_id,
                per_worker,
                min_text_len,
                tmp_path,
            ),
            daemon = True,
        )
        p.start()
        processes.append(p)
        print(f"  Started worker {worker_id} "
              f"({len(file_chunks[worker_id])} files)", flush=True)

    # Wait for all workers to finish
    results       = []
    total_written = 0
    total_skipped = 0

    for _ in range(actual_workers):
        result = result_queue.get()
        results.append(result)
        total_written += result["written"]
        total_skipped += result["skipped"]
        print(
            f"  [worker {result['worker_id']}] ✓ "
            f"{result['written']:,} written, "
            f"{result['skipped']:,} skipped",
            flush=True,
        )

    for p in processes:
        p.join(timeout=30)

    print(f"\n  Total written : {total_written:,}")
    print(f"  Total skipped : {total_skipped:,}")

    return [r["tmp_path"] for r in results if r["written"] > 0]


# ─────────────────────────────────────────────────────────────
# Stage 2 — BPE training from files
# ─────────────────────────────────────────────────────────────

def _build_tokenizer(
    text_files:    List[str],
    vocab_size:    int,
    min_frequency: int,
) -> Tokenizer:
    """
    Train BPE from a list of text files.
    Rust backend parallelizes word frequency counting across files.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))

    tokenizer.normalizer = NormSequence([
        Replace("\u2018", "'"),
        Replace("\u2019", "'"),
        Replace("\u201c", '"'),
        Replace("\u201d", '"'),
        Replace("\u2013", "-"),
        Replace("\u2014", "-"),
        NFKC(),
        Strip(),
    ])

    tokenizer.pre_tokenizer = Sequence([
        Digits(individual_digits=True),   # "1234" → "1","2","3","4"
        Punctuation(),
        ByteLevel(add_prefix_space=False),
    ])

    trainer = BpeTrainer(
        vocab_size                = vocab_size,
        min_frequency             = min_frequency,
        special_tokens            = SPECIAL_TOKENS,
        initial_alphabet          = ByteLevel.alphabet(),  # all 256 bytes guaranteed
        continuing_subword_prefix = "",                    # Llama/GPT-NeoX convention
        show_progress             = True,
        # limit_alphabet intentionally NOT set — breaks ByteLevel coverage
    )

    print(f"\n[Stage 2] Training BPE from {len(text_files)} files...")
    print("  Rust backend parallelizes word frequency counting across files")

    # train() from files: memory-mapped, Rust-parallelized
    # Much faster and more memory efficient than train_from_iterator
    tokenizer.train(text_files, trainer)

    return tokenizer


# ─────────────────────────────────────────────────────────────
# Post-training checks
# ─────────────────────────────────────────────────────────────

def _verify_tokenizer(tokenizer: Tokenizer):
    """Verify token IDs and byte fallback coverage."""

    # 1. Token ID check
    actual   = {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    expected = {"<pad>": PAD_ID, "<eos>": EOS_ID, "<bos>": BOS_ID, "<unk>": UNK_ID}
    assert actual == expected, (
        f"Token ID mismatch!\nExpected: {expected}\nGot: {actual}"
    )
    print(f"  ✓ Special token IDs: {actual}")

    # 2. Byte fallback — <unk> should never appear for any valid input
    unk_id = tokenizer.token_to_id("<unk>")
    test_strings = [
        "Hello world! 1+2=3",
        "∫∑√≤≥∈∉⊂⊃",
        "def fibonacci(n): return n",
        "中文日本語한국어",
        r"\frac{a}{b} = \sqrt{c^2}",
        "🎉🔥💡",
        "\x00\x01\x7f\xff",
    ]
    failures = [s for s in test_strings if unk_id in tokenizer.encode(s).ids]
    if failures:
        print(f"  ⚠ <unk> found in {len(failures)} test cases: {failures}")
    else:
        print("  ✓ No <unk> tokens — byte fallback verified")

    # 3. Vocab report
    total = tokenizer.get_vocab_size()
    print(f"  ✓ Vocab: {total:,}  MXU aligned: {total % 128 == 0}")


def _save_tokenizer(tokenizer: Tokenizer, output_path: str):
    """Save tokenizer.json + tokenizer_config.json + special_tokens_map.json."""
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tokenizer.save(output_path)

    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class":              "PreTrainedTokenizerFast",
            "model_max_length":             2048,
            "bos_token":                    "<bos>",
            "eos_token":                    "<eos>",
            "unk_token":                    "<unk>",
            "pad_token":                    "<pad>",
            "padding_side":                 "right",
            "truncation_side":              "right",
            "add_bos_token":                False,
            "add_eos_token":                False,
            "clean_up_tokenization_spaces": False,
        }, f, indent=2)

    with open(os.path.join(out_dir, "special_tokens_map.json"), "w") as f:
        json.dump({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }, f, indent=2)

    print(f"  ✓ Saved → {output_path}")


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def train_tokenizer(
    dataset_name:  str,
    vocab_size:    int = 32_000,
    output_path:   str = "tokenizer/tokenizer.json",
    min_frequency: int = 2,
    max_samples:   int = 5_000_000,
    min_text_len:  int = 50,
    num_workers:   int = 4,
    work_dir:      str = "/tmp/tok_work",
    cleanup:       bool = True,
):
    """
    Train a BPE tokenizer in two stages:
      1. Parallel file-level data collection → temp text files
      2. File-based BPE training → Rust parallelism

    Args:
        dataset_name:  HF repo ID of your mixed tokenizer dataset.
        vocab_size:    must be divisible by 128. Default 32_000.
        output_path:   where to save tokenizer.json.
        min_frequency: BPE merge threshold. Use 2 for 5M+ samples.
        max_samples:   total samples to collect. Default 5_000_000.
        min_text_len:  skip texts shorter than this (chars).
        num_workers:   parallel workers. Auto-capped to cpu_count().
                       Each worker gets its own parquet files — no duplicates.
        work_dir:      temp directory for intermediate text files.
        cleanup:       delete temp files after training. Default True.
    """
    import time
    _validate_vocab_size(vocab_size)
    t0 = time.time()

    print("=" * 60)
    print("LaughLM Tokenizer Training")
    print(f"  Dataset      : {dataset_name}")
    print(f"  Vocab size   : {vocab_size:,}")
    print(f"  Max samples  : {max_samples:,}")
    print(f"  Min frequency: {min_frequency}")
    print(f"  Num workers  : {min(num_workers, cpu_count())} "
          f"(requested {num_workers}, avail {cpu_count()})")
    print("=" * 60)

    # ── Stage 1: Collect data ─────────────────────────────────
    t1 = time.time()
    text_files = _collect_to_files(
        dataset_name  = dataset_name,
        total_samples = max_samples,
        num_workers   = num_workers,
        min_text_len  = min_text_len,
        work_dir      = work_dir,
    )
    print(f"\n  Stage 1 done in {time.time()-t1:.1f}s")

    if not text_files:
        raise RuntimeError("No data collected — check dataset_name and HF credentials")

    # ── Stage 2: Train BPE ────────────────────────────────────
    t2 = time.time()
    tokenizer = _build_tokenizer(text_files, vocab_size, min_frequency)
    print(f"\n  Stage 2 done in {time.time()-t2:.1f}s")

    # ── Verify + save ─────────────────────────────────────────
    print("\nVerifying tokenizer...")
    _verify_tokenizer(tokenizer)
    _save_tokenizer(tokenizer, output_path)

    # ── Cleanup temp files ────────────────────────────────────
    if cleanup:
        for f in text_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    total = time.time() - t0
    print(f"\n── Training Complete ─────────────────────────────────")
    print(f"  Total time : {total:.1f}s ({total/60:.1f} min)")
    print(f"  Output     : {output_path}")
    print("─────────────────────────────────────────────────────")

    return tokenizer
