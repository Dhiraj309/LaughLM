import json
import os
import random
import time
from typing import List
from multiprocessing import Process, Queue, cpu_count

from datasets import load_dataset
from huggingface_hub import HfApi
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import (
    NFKC,
    Replace,
    Sequence as NormSequence,
    Strip,
)
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# ── CPU SETTINGS ─────────────────────────────────────────────
NUM_CPUS = cpu_count()
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAYON_RS_NUM_CPUS"] = str(NUM_CPUS)

# ── Special tokens ───────────────────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]
PAD_ID, EOS_ID, BOS_ID, UNK_ID = 0, 1, 2, 3

# Prevent LaTeX/code truncation
_MAX_WORD_LEN = 200


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────

def _validate_vocab_size(vocab_size: int):
    assert vocab_size % 128 == 0, "vocab_size must be divisible by 128"


def _truncate_long_words(text: str, max_len: int = _MAX_WORD_LEN) -> str:
    return " ".join(
        w if len(w) <= max_len else w[:max_len]
        for w in text.split()
    )


def _get_parquet_files(dataset_name: str) -> List[str]:
    api = HfApi()
    all_files = api.list_repo_files(dataset_name, repo_type="dataset")
    parquet_files = sorted(f for f in all_files if f.endswith(".parquet"))
    print(f"Found {len(parquet_files)} parquet files")
    return parquet_files


def _split_files(files: List[str], num_workers: int) -> List[List[str]]:
    chunks = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        chunks[i % num_workers].append(f)
    return chunks


# ─────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────

def _producer_worker(
    result_queue: Queue,
    dataset_name: str,
    file_list: List[str],
    worker_id: int,
    per_worker: int,
    min_text_len: int,
    tmp_path: str,
):
    written = 0
    skipped = 0
    buffer = []

    try:
        with open(tmp_path, "w", encoding="utf-8", buffering=4 * 1024 * 1024) as f:
            for filename in file_list:
                if written >= per_worker:
                    break

                ds = load_dataset(
                    dataset_name,
                    split="train",
                    data_files=filename,
                    streaming=True,
                )

                for sample in ds:
                    if written >= per_worker:
                        break

                    text = sample.get("text", "")
                    if not text or len(text) < min_text_len:
                        skipped += 1
                        continue

                    text = _truncate_long_words(text)
                    buffer.append(text.replace("\n", " ").strip())
                    written += 1

                    # Shuffle buffer to avoid domain-order bias
                    if len(buffer) >= 10000:
                        random.shuffle(buffer)
                        for t in buffer:
                            f.write(t + "\n")
                        buffer = []

            # Flush remaining buffer
            if buffer:
                random.shuffle(buffer)
                for t in buffer:
                    f.write(t + "\n")

    except Exception as e:
        print(f"[worker {worker_id}] error: {e}", flush=True)

    finally:
        result_queue.put({
            "worker_id": worker_id,
            "tmp_path": tmp_path,
            "written": written,
            "skipped": skipped,
        })


# ─────────────────────────────────────────────────────────────
# Stage 1 — Collect text
# ─────────────────────────────────────────────────────────────

def _collect_to_files(
    dataset_name: str,
    total_samples: int,
    num_workers: int,
    min_text_len: int,
    work_dir: str,
) -> List[str]:

    parquet_files = _get_parquet_files(dataset_name)
    file_chunks = _split_files(parquet_files, num_workers)
    per_worker = (total_samples + num_workers - 1) // num_workers

    print(f"\n[Stage 1] Collecting {total_samples:,} samples")

    os.makedirs(work_dir, exist_ok=True)

    result_queue = Queue()
    processes = []

    for worker_id in range(num_workers):
        tmp_path = os.path.join(work_dir, f"worker_{worker_id:02d}.txt")
        p = Process(
            target=_producer_worker,
            args=(
                result_queue,
                dataset_name,
                file_chunks[worker_id],
                worker_id,
                per_worker,
                min_text_len,
                tmp_path,
            ),
        )
        p.start()
        processes.append(p)

    results = []
    for _ in range(num_workers):
        result = result_queue.get()
        results.append(result)
        print(f"Worker {result['worker_id']}: {result['written']:,} written")

    for p in processes:
        p.join()

    return [r["tmp_path"] for r in results if r["written"] > 0]


# ─────────────────────────────────────────────────────────────
# Stage 2 — Train tokenizer
# ─────────────────────────────────────────────────────────────

def _build_tokenizer(text_files, vocab_size, min_frequency):

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

    # Frontier-style ByteLevel BPE
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        continuing_subword_prefix="",
        show_progress=True,
    )

    tokenizer.train(text_files, trainer)
    return tokenizer


# ─────────────────────────────────────────────────────────────
# Verify + Save
# ─────────────────────────────────────────────────────────────

def _verify_tokenizer(tokenizer):
    actual = {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS}
    expected = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3}
    assert actual == expected, f"Token ID mismatch! {actual}"

    print("✓ Special tokens verified")
    print("✓ Vocab size:", tokenizer.get_vocab_size())

    unk_id = tokenizer.token_to_id("<unk>")
    test_strings = [
        "Hello world!",
        "∫∑√≤≥",
        "def func(x): return x",
        "中文测试",
        "🎉🔥",
        "\x00\x01\x02",
    ]

    for s in test_strings:
        if unk_id in tokenizer.encode(s).ids:
            print("⚠ UNK found in:", s)
            return

    print("✓ Byte fallback working (no UNK)")


def _save_tokenizer(tokenizer, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)

    with open(os.path.join(os.path.dirname(output_path), "tokenizer_config.json"), "w") as f:
        json.dump({
            "tokenizer_class": "PreTrainedTokenizerFast",
            "model_max_length": 2048,
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "padding_side": "right",
            "truncation_side": "right",
            "add_bos_token": False,
            "add_eos_token": False,
            "clean_up_tokenization_spaces": False,
        }, f, indent=2)

    with open(os.path.join(os.path.dirname(output_path), "special_tokens_map.json"), "w") as f:
        json.dump({
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }, f, indent=2)

    print(f"✓ Saved tokenizer → {output_path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def train_tokenizer(
    dataset_name: str,
    vocab_size: int = 32000,
    output_path: str = "/tmp/tokenizer.json",
    min_frequency: int = 5,
    max_samples: int = 18_500_000,
    min_text_len: int = 50,
    num_workers: int = 32,
    work_dir: str = "/tmp/tok_work",
):
    random.seed(42)

    _validate_vocab_size(vocab_size)
    t0 = time.time()

    text_files = _collect_to_files(
        dataset_name,
        max_samples,
        num_workers,
        min_text_len,
        work_dir,
    )

    tokenizer = _build_tokenizer(text_files, vocab_size, min_frequency)
    _verify_tokenizer(tokenizer)
    _save_tokenizer(tokenizer, output_path)

    print(f"\nTraining finished in {(time.time()-t0)/60:.2f} min")
    return tokenizer
