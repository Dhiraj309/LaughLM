from __future__ import annotations

import math
import argparse
from dataclasses import replace

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


def make_parity_config(llama_config):
    """
    Make native JAX comparable to HF for export parity.

    Do not use Splash/bf16 here. This is not a training benchmark.
    """

    return replace(
        llama_config,
        attention_impl="xla",
        attention_fallback="warn",
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


def torch_ce_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].float()
    shift_labels = input_ids[:, 1:]

    return torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        reduction="mean",
    )


def jax_ce_loss(
    logits: jnp.ndarray,
    input_ids: jnp.ndarray,
) -> jnp.ndarray:
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


def load_token_batch(args):
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

    return input_ids_np


def restore_params(checkpoint_dir, exp_config):
    print("\n================ RESTORE JAX CHECKPOINT ================\n")

    checkpoints = CheckpointManager(
        checkpoint_dir,
    )

    backend = str(
        getattr(
            exp_config.runtime,
            "canonical_backend",
            exp_config.runtime.backend,
        )
    )

    if backend == "pmap":
        num_devices = int(jax.local_device_count())

    elif backend == "fsdp":
        raise NotImplementedError(
            "debug_hf_export_loss_parity.py cannot restore FSDP checkpoints "
            "directly yet. Use the Phase 4B canonical unshard/gather export "
            "path first."
        )

    else:
        raise NotImplementedError(
            f"HF parity debug restore for backend={backend!r} is not implemented."
        )

    train_llama_config = build_llama_config(
        exp_config
    )

    target_model = LlamaForCausalLM(
        config=train_llama_config
    )

    rng = jax.random.PRNGKey(0)

    dummy = jnp.zeros(
        (
            exp_config.runtime.micro_batch_per_device,
            exp_config.runtime.seq_len,
        ),
        dtype=jnp.int32,
    )

    variables = target_model.init(
        rng,
        input_ids=dummy,
        use_cache=False,
        mode="train",
        return_hidden=bool(
            exp_config.architecture.weight_tying
        ),
    )

    from LaughLM.training.optimizer import build_optimizer
    from LaughLM.training.scheduler import build_scheduler

    schedule = build_scheduler(
        exp_config,
        num_devices=num_devices,
    )

    optimizer = build_optimizer(
        exp_config,
        schedule,
    )

    target_state = TrainState(
        params=variables["params"],
        opt_state=optimizer.init(variables["params"]),
        step=jnp.asarray(0, dtype=jnp.int32),
        tokens_processed=jnp.asarray(0, dtype=jnp.int64),
        rng_key=rng,
    )

    restored = checkpoints.restore_latest(
        target_state=target_state,
        config=exp_config,
        num_devices=num_devices,
        require_metadata=True,
        require_v3=True,
        purpose="hf_parity_debug",
    )



    

    if restored is None:
        raise RuntimeError("No checkpoint found.")

    state, step = restored

    print("restored step:", step)

    if isinstance(state, TrainState):
        params = state.params
    else:
        params = state["params"]

    params = unbox_logically_partitioned(
        params
    )

    return params, step


def run_native(
    *,
    params,
    llama_config,
    input_ids_np,
):
    print("\n================ NATIVE JAX LOSS ================\n")
    print("native attention_impl:", llama_config.attention_impl)
    print("native compute_dtype:", llama_config.compute_dtype)
    print("native output_dtype:", llama_config.output_dtype)

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

    native_loss_value = float(
        jax.device_get(native_loss)
    )

    native_ppl = math.exp(
        native_loss_value
    )

    print("native loss:", native_loss_value)
    print("native ppl: ", native_ppl)

    native_logits_np = np.asarray(
        jax.device_get(native_logits),
        dtype=np.float32,
    )

    return native_logits_np, native_loss_value


def run_hf(
    *,
    hf_dir,
    input_ids_np,
):
    print("\n================ HF LOSS ================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        hf_dir,
        use_fast=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
    except TypeError:
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_dir,
            torch_dtype=torch.float32,
            device_map="auto",
            attn_implementation="eager",
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

    hf_loss_value = float(
        hf_loss.detach().cpu()
    )

    hf_ppl = math.exp(
        hf_loss_value
    )

    print("hf loss:", hf_loss_value)
    print("hf ppl: ", hf_ppl)

    hf_logits_np = (
        hf_logits
        .detach()
        .cpu()
        .float()
        .numpy()
    )

    return hf_logits_np, hf_loss_value


def report_parity(
    *,
    native_logits_np,
    hf_logits_np,
    native_loss_value,
    hf_loss_value,
):
    print("\n================ LOGIT PARITY ================\n")

    diff = np.abs(
        native_logits_np - hf_logits_np,
    )

    print("native logits shape:", native_logits_np.shape)
    print("hf logits shape:    ", hf_logits_np.shape)
    print("max abs diff:", float(diff.max()))
    print("mean abs diff:", float(diff.mean()))
    print("p50 abs diff:", float(np.percentile(diff, 50)))
    print("p95 abs diff:", float(np.percentile(diff, 95)))
    print("p99 abs diff:", float(np.percentile(diff, 99)))

    loss_delta = abs(
        native_loss_value - hf_loss_value
    )

    print("\n================ DIAGNOSIS ================\n")
    print("loss delta:", loss_delta)

    if loss_delta < 0.05 and float(diff.mean()) < 0.05:
        print("HF export matches native checkpoint closely enough for practical validation.")
    elif native_loss_value < 4.0 and hf_loss_value > 5.0:
        print("JAX checkpoint is good, HF export/path is still wrong.")
        print("Because this script uses XLA fp32 native and fp32 HF, inspect RoPE/QK layout or parameter mapping.")
    elif native_loss_value > 5.0 and hf_loss_value > 5.0:
        print("Both JAX and HF are bad on this shard under parity config.")
        print("Check checkpoint/config/data alignment.")
    else:
        print("Partial mismatch.")
        print("Inspect mean/p99 logit diff and compare seq_len sweep.")


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

    exp_config = load_config(
        args.config
    )

    train_llama_config = build_llama_config(
        exp_config
    )

    llama_config = make_parity_config(
        train_llama_config
    )

    print("\n================ CONFIG ================\n")
    print("hidden_size:", llama_config.hidden_size)
    print("layers:", llama_config.num_hidden_layers)
    print("heads:", llama_config.num_attention_heads)
    print("kv_heads:", llama_config.num_key_value_heads)
    print("vocab:", llama_config.vocab_size)
    print("bos:", llama_config.bos_token_id)
    print("eos:", llama_config.eos_token_id)
    print("pad:", llama_config.pad_token_id)
    print("attention_impl:", llama_config.attention_impl)
    print("compute_dtype:", llama_config.compute_dtype)
    print("output_dtype:", llama_config.output_dtype)

    input_ids_np = load_token_batch(
        args
    )

    params, _ = restore_params(
        args.checkpoint_dir,
        exp_config,
    )

    native_logits_np, native_loss_value = run_native(
        params=params,
        llama_config=llama_config,
        input_ids_np=input_ids_np,
    )

    hf_logits_np, hf_loss_value = run_hf(
        hf_dir=args.hf_dir,
        input_ids_np=input_ids_np,
    )

    report_parity(
        native_logits_np=native_logits_np,
        hf_logits_np=hf_logits_np,
        native_loss_value=native_loss_value,
        hf_loss_value=hf_loss_value,
    )


if __name__ == "__main__":
    main()
