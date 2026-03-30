  
import json  
import os  
from typing import Iterable, List, Optional  
from multiprocessing import Process, Queue, cpu_count  
  
from datasets import load_dataset  
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
  
# ── MAX CPU UTILIZATION ──────────────────────────────────────  
os.environ["TOKENIZERS_PARALLELISM"] = "true"  
os.environ["RAYON_RS_NUM_CPUS"] = str(cpu_count())  
  
# ── Special tokens ───────────────────────────────────────────  
SPECIAL_TOKENS = ["<pad>", "<eos>", "<bos>", "<unk>"]  
PAD_ID, EOS_ID, BOS_ID, UNK_ID = 0, 1, 2, 3  
  
_MAX_WORD_LEN = 50  
  
  
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
  
  
# ─────────────────────────────────────────────────────────────  
# 🚀 Parallel streaming iterator  
# ─────────────────────────────────────────────────────────────  
  
def _producer_worker(queue, dataset_name, max_samples, min_text_len, worker_id):  
    ds = load_dataset(dataset_name, split="train", streaming=True)  
  
    for i, sample in enumerate(ds):  
        if max_samples and i >= max_samples:  
            break  
  
        text = sample.get("text", "")  
        if not text or len(text) < min_text_len:  
            continue  
  
        text = _truncate_long_words(text)  
        queue.put(text)  
  
    queue.put(None)  
  
  
def hf_text_batch_iterator_parallel(  
    dataset_name: str,  
    batch_size: int = 10_000,  
    max_samples: Optional[int] = None,  
    min_text_len: int = 50,  
    num_workers: int = 4,  
) -> Iterable[List[str]]:  
  
    queue = Queue(maxsize=50_000)  
  
    workers = []  
    for i in range(num_workers):  
        p = Process(  
            target=_producer_worker,  
            args=(queue, dataset_name, max_samples, min_text_len, i),  
        )  
        p.start()  
        workers.append(p)  
  
    batch = []  
    finished_workers = 0  
    yielded = 0  
  
    while True:  
        item = queue.get()  
  
        if item is None:  
            finished_workers += 1  
            if finished_workers == num_workers:  
                break  
            continue  
  
        batch.append(item)  
        yielded += 1  
  
        if len(batch) >= batch_size:  
            yield batch  
            batch = []  
  
        if yielded % 100_000 == 0:  
            print(f" [iterator] {yielded:,} samples", flush=True)  
  
    if batch:  
        yield batch  
  
    for p in workers:  
        p.join()  
  
    print(f" [iterator] done — {yielded:,} total samples", flush=True)  
  
  
# ─────────────────────────────────────────────────────────────  
# Training  
# ─────────────────────────────────────────────────────────────  
  
def train_tokenizer(  
    dataset_name: str,  
    vocab_size: int = 32_000,  
    output_path: str = "tokenizer/tokenizer.json",  
    min_frequency: int = 2,  
    batch_size: int = 10_000,  
    max_samples: Optional[int] = None,  
    min_text_len: int = 50,  
    num_workers: int = 4,  
):  
  
    _validate_vocab_size(vocab_size)  
  
    print("\n── Optimized Tokenizer Training ─────────────────────")  
    print(f" Dataset : {dataset_name}")  
    print(f" Batch size : {batch_size:,}")  
    print(f" Workers : {num_workers}")  
    print(f" Max samples : {max_samples}")  
    print("─────────────────────────────────────────────────────\n")  
  
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))  
  
    # Normalizer  
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
  
    # Pre-tokenizer  
    tokenizer.pre_tokenizer = Sequence([  
        Digits(individual_digits=True),  
        Punctuation(),  
        ByteLevel(add_prefix_space=False),  
    ])  
  
    # Trainer  
    trainer = BpeTrainer(  
        vocab_size=vocab_size,  
        min_frequency=min_frequency,  
        special_tokens=SPECIAL_TOKENS,  
        initial_alphabet=ByteLevel.alphabet(),  
        show_progress=True,  
        continuing_subword_prefix="",  
    )  
  
    print("🚀 Training started...")  
  
    tokenizer.train_from_iterator(  
        hf_text_batch_iterator_parallel(  
            dataset_name=dataset_name,  
            batch_size=batch_size,  
            max_samples=max_samples,  
            min_text_len=min_text_len,  
            num_workers=num_workers,  
        ),  
        trainer=trainer,  
    )  
  
    # Save  
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)  
    tokenizer.save(output_path)  
  
    print("\n✅ Training complete!")  
    print(f" Vocab size: {tokenizer.get_vocab_size():,}")  
  
    return tokenizer
