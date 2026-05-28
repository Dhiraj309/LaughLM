from __future__ import annotations

import math
import argparse

import numpy as np
import torch

import jax
import jax.numpy as jnp

from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.train_state import TrainState
from LaughLM.export.validate_hf import unbox_logically_partitioned


def torch_ce_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]

    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="mean",
    )


def jax_ce_loss(logits: jnp.ndarray, input_ids: jnp.ndarray) -> jnp.ndarray:
    shift_logits = logits[:, :-1, :].astype(jnp.float32)
    shift_labels = input_ids[:, 1:]

    log_probs = jax.nn.log_softmax(
        shift_logits,
        axis=-1,
    )

    token_log_probs = jnp.take_along_axis(
        log_probs,
        shift_labels[..., None],
        axis=-1,
    ).squeeze(-1)

    return -jnp.mean(token_log_probs)


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
        "--hf_dir",
        required=True,
        help="Local HF export dir or HF repo id.",
    )

    parser.add_argument(
        "--dataset_repo",
        default="LaughTaleAI/LaughLM-Tokenized-Fine",
    )

    parser.add_argument(
        "--dataset_file",
        default="fineweb-edu/fineweb-edu_shard_00000.bin",
    )

    parser.add_argument(
        "--offset",
        type=int,
        default=50_000_000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
    )

    args = parser.parse_args()

    # ============================================================
    # Load exact training config
    # ============================================================

    exp_config = load_config(args.config)
    llama_config = build_llama_config(exp_config)

    print("\n================ CONFIG ================\n")
    print("hidden_size:", llama_config.hidden_size)
    print("layers:", llama_config.num_hidden_layers)
    print("heads:", llama_config.num_attention_heads)
    print("kv_heads:", llama_config.num_key_value_heads)
    print("vocab:", llama_config.vocab_size)
    print("bos:", llama_config.bos_token_id)
    print("eos:", llama_config.eos_token_id)
    print("pad:", llama_config.pad_token_id)

    # ============================================================
    # Load token shard
    # ============================================================

    bin_path = hf_hub_download(
        repo_id=args.dataset_repo,
        filename=args.dataset_file,
        repo_type="dataset",
    )

    data = np.memmap(
        bin_path,
        dtype=np.uint16,
        mode="r",
    )

    n_tokens = args.batch_size * args.seq_len

    chunk = np.asarray(
        data[args.offset : args.offset + n_tokens],
        dtype=np.int64,
    )

    input_ids_np = chunk.reshape(
        args.batch_size,
        args.seq_len,
    )

    print("\n================ DATA ================\n")
    print("bin path:", bin_path)
    print("data tokens:", len(data))
    print("offset:", args.offset)
    print("batch:", input_ids_np.shape)
    print("min token:", int(input_ids_np.min()))
    print("max token:", int(input_ids_np.max()))

    # ============================================================
    # Restore native JAX checkpoint
    # ============================================================

    print("\n================ RESTORE JAX CHECKPOINT ================\n")

    checkpoints = CheckpointManager(
        args.checkpoint_dir,
    )

    restored = checkpoints.restore_latest(
        target_state=None,
    )

    if restored is None:
        raise RuntimeError("No checkpoint found.")

    state, step = restored

    print("restored step:", step)

    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state["params"]

    params = unbox_logically_partitioned(params)

    # ============================================================
    # Native JAX forward/loss
    # ============================================================

    print("\n================ NATIVE JAX LOSS ================\n")

    native_model = LlamaForCausalLM(
        config=llama_config,
    )

    input_ids_jax = jnp.asarray(
        input_ids_np,
        dtype=jnp.int32,
    )

    native_logits, _ = native_model.apply(
        {"params": params},
        input_ids=input_ids_jax,
        use_cache=False,
        mode="train",
    )

    native_loss = jax_ce_loss(
        native_logits,
        input_ids_jax,
    )

    native_loss_value = float(jax.device_get(native_loss))
    native_ppl = math.exp(native_loss_value)

    print("native loss:", native_loss_value)
    print("native ppl: ", native_ppl)

    native_logits_np = np.asarray(
        jax.device_get(native_logits),
        dtype=np.float32,
    )

    # ============================================================
    # HF forward/loss
    # ============================================================

    print("\n================ HF LOSS ================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_dir,
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    hf_model.eval()

    hf_model.config.bos_token_id = 1
    hf_model.config.eos_token_id = 32000
    hf_model.config.pad_token_id = 32000

    if hasattr(hf_model, "generation_config"):
        hf_model.generation_config.bos_token_id = 1
        hf_model.generation_config.eos_token_id = 32000
        hf_model.generation_config.pad_token_id = 32000

    input_ids_torch = torch.tensor(
        input_ids_np,
        dtype=torch.long,
        device=hf_model.device,
    )

    with torch.inference_mode():
        hf_outputs = hf_model(
            input_ids=input_ids_torch,
            use_cache=False,
        )

        hf_logits = hf_outputs.logits
        hf_loss = torch_ce_loss(
            hf_logits,
            input_ids_torch,
        )

    hf_loss_value = float(hf_loss.detach().cpu())
    hf_ppl = math.exp(hf_loss_value)

    print("hf loss:", hf_loss_value)
    print("hf ppl: ", hf_ppl)

    hf_logits_np = hf_logits.detach().cpu().float().numpy()

    # ============================================================
    # Logit parity
    # ============================================================

    print("\n================ LOGIT PARITY ================\n")

    diff = np.abs(
        native_logits_np - hf_logits_np,
    )

    print("native logits shape:", native_logits_np.shape)
    print("hf logits shape:    ", hf_logits_np.shape)
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("p99 abs diff:", float(np.percentile(diff, 99)))

    print("\n================ DIAGNOSIS ================\n")

    if native_loss_value < 4.0 and hf_loss_value > 5.0:
        print("JAX checkpoint is good, HF export is wrong.")
        print("Focus on convert_params.py: QKV split, MLP gate/up/down, transposes, RoPE layout.")
    elif native_loss_value > 5.0 and hf_loss_value > 5.0:
        print("Both JAX and HF are bad on this shard.")
        print("Then the training log loss may be computed/reported differently or wrong checkpoint was restored.")
    elif abs(native_loss_value - hf_loss_value) < 0.1:
        print("HF export matches native checkpoint.")
        print("Generation weakness is from model undertraining, not export.")
    else:
        print("Partial mismatch. Inspect max/mean logit diff and parameter mapping.")


if __name__ == "__main__":
    main()
