"""
LaughLM/export/export_hf.py

Export a LaughLM checkpoint to Hugging Face LlamaForCausalLM format.

Safety policy
-------------
- Restore requires v3 checkpoint metadata.
- Restore uses a real LLaMA TrainState target, not target_state=None.
- PMAP native params can be exported directly.
- FSDP export is intentionally blocked until canonical unshard/gather
  support is implemented.
"""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp

from safetensors.numpy import save_file
from transformers import GenerationConfig

from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler
from LaughLM.training.train_state import TrainState
from LaughLM.utils.rng import create_rng

from LaughLM.export.convert_params import (
    convert_params_to_hf,
    validate_exported_tensors,
)

from LaughLM.export.hf_config import build_hf_config
from LaughLM.export.validate_hf import validate_hf_export


# ============================================================
# Backend helpers
# ============================================================

def _canonical_backend(config) -> str:
    return str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )


def _metadata_num_devices(config) -> int:
    backend = _canonical_backend(config)

    if backend == "pmap":
        return int(jax.local_device_count())

    if backend == "fsdp":
        return int(
            config.spmd.mesh.axis_sizes().get(
                "data",
                1,
            )
        )

    raise ValueError(
        "HF export only supports implemented training backends.\n"
        f"  runtime.backend={config.runtime.backend!r}\n"
        f"  canonical_backend={backend!r}"
    )


def _require_supported_export_backend(config) -> None:
    backend = _canonical_backend(config)

    if backend == "pmap":
        return

    if backend == "fsdp":
        raise NotImplementedError(
            "FSDP HF export is not enabled yet. "
            "FSDP checkpoints require canonical unshard/gather support "
            "before convert_params_to_hf() can safely consume params."
        )

    raise NotImplementedError(
        "Reserved backend cannot be exported to HF yet.\n"
        f"  runtime.backend={config.runtime.backend!r}\n"
        f"  canonical_backend={backend!r}"
    )


# ============================================================
# Restore target
# ============================================================

def _build_target_state(
    config,
    *,
    num_devices: int,
) -> TrainState:
    llama_config = build_llama_config(
        config
    )

    model = LlamaForCausalLM(
        config=llama_config
    )

    rng = create_rng(
        seed=0
    )

    dummy = jnp.zeros(
        (
            config.runtime.micro_batch_per_device,
            config.runtime.seq_len,
        ),
        dtype=jnp.int32,
    )

    variables = model.init(
        rng.next_key(),
        input_ids=dummy,
        use_cache=False,
        mode="train",
        return_hidden=bool(
            config.architecture.weight_tying
        ),
    )

    params = variables["params"]

    schedule = build_scheduler(
        config,
        num_devices=num_devices,
    )

    optimizer = build_optimizer(
        config,
        schedule,
    )

    opt_state = optimizer.init(
        params
    )

    return TrainState(
        params=params,
        opt_state=opt_state,
        step=jnp.asarray(
            0,
            dtype=jnp.int32,
        ),
        tokens_processed=jnp.asarray(
            0,
            dtype=jnp.int64,
        ),
        rng_key=rng.key,
    )


# ============================================================
# Tokenizer copy
# ============================================================

def copy_tokenizer_files(
    source_dir,
    output_dir,
) -> None:
    source_dir = Path(
        source_dir
    )

    output_dir = Path(
        output_dir
    )

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ]

    copied = []

    for filename in tokenizer_files:
        src = source_dir / filename

        if src.exists():
            dst = output_dir / filename

            shutil.copy2(
                src,
                dst,
            )

            copied.append(
                filename
            )

    if not copied:
        raise RuntimeError(
            "No tokenizer files found.\n"
            f"source_dir={source_dir}"
        )

    print(
        "[export] copied tokenizer files:"
    )

    for filename in copied:
        print(
            f"  - {filename}"
        )


# ============================================================
# Generation config
# ============================================================

def save_generation_config(
    output_dir,
    llama_config,
) -> None:
    generation_config = GenerationConfig(
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        pad_token_id=llama_config.pad_token_id,
        max_length=llama_config.max_position_embeddings,
        do_sample=False,
        use_cache=True,
    )

    generation_config.save_pretrained(
        output_dir
    )

    print(
        "[export] saved generation config"
    )


# ============================================================
# Config save
# ============================================================

def save_hf_config(
    output_dir,
    hf_config,
) -> None:
    output_dir = Path(
        output_dir
    )

    config_path = (
        output_dir
        / "config.json"
    )

    with open(
        config_path,
        "w",
    ) as f:
        json.dump(
            hf_config,
            f,
            indent=2,
        )

    print(
        "[export] saved config.json"
    )


# ============================================================
# Main export
# ============================================================

def export_hf_checkpoint(
    *,
    config_path,
    checkpoint_dir,
    output_dir,
    tokenizer_dir,
    validate=True,
) -> None:
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "[export] loading config..."
    )

    exp_config = load_config(
        config_path
    )

    backend = _canonical_backend(
        exp_config
    )

    num_devices = _metadata_num_devices(
        exp_config
    )

    print(
        "[export] backend:\n"
        f"  raw={exp_config.runtime.backend}\n"
        f"  canonical={backend}\n"
        f"  metadata_num_devices={num_devices}"
    )

    _require_supported_export_backend(
        exp_config
    )

    llama_config = build_llama_config(
        exp_config
    )

    print(
        "[export] building restore target..."
    )

    target_state = _build_target_state(
        exp_config,
        num_devices=num_devices,
    )

    print(
        "[export] restoring checkpoint..."
    )

    checkpoints = CheckpointManager(
        checkpoint_dir
    )

    try:
        restored = checkpoints.restore_latest(
            target_state=target_state,
            config=exp_config,
            num_devices=num_devices,
            require_metadata=True,
            require_v3=True,
            purpose="hf_export",
        )

        if restored is None:
            raise RuntimeError(
                "No checkpoint found."
            )

        state, step = restored

        print(
            f"[export] restored step={step:,}"
        )

        params = jax.device_get(
            state.params
        )

        source_metadata = checkpoints.load_metadata(
            step
        )

        print(
            "[export] converting tensors..."
        )

        tensors = convert_params_to_hf(
            params=params,
            config=llama_config,
        )

        validate_exported_tensors(
            tensors
        )

        total_tensors = len(
            tensors
        )

        total_params = sum(
            tensor.size
            for tensor in tensors.values()
        )

        print(
            f"[export] converted {total_tensors:,} tensors"
        )

        print(
            f"[export] total params: {total_params:,}"
        )

        del state
        gc.collect()

        print(
            "[export] saving safetensors..."
        )

        safetensor_path = (
            output_dir
            / "model.safetensors"
        )

        metadata = {
            "format": "pt",
            "framework": "huggingface",
            "source": "LaughLM",
            "backend": backend,
            "checkpoint_step": str(
                int(step)
            ),
        }

        save_file(
            tensors,
            str(safetensor_path),
            metadata=metadata,
        )

        print(
            "[export] saved model.safetensors"
        )

        print(
            "[export] building HF config..."
        )

        hf_config = build_hf_config(
            llama_config
        )

        save_hf_config(
            output_dir,
            hf_config,
        )

        save_generation_config(
            output_dir,
            llama_config,
        )

        print(
            "[export] copying tokenizer..."
        )

        copy_tokenizer_files(
            tokenizer_dir,
            output_dir,
        )

        source_metadata_path = (
            output_dir
            / "source_checkpoint_metadata.json"
        )

        with open(
            source_metadata_path,
            "w",
        ) as f:
            json.dump(
                source_metadata,
                f,
                indent=2,
                sort_keys=True,
            )

        print(
            "[export] saved source checkpoint metadata"
        )

        if validate:
            print(
                "[export] running validation..."
            )

            validate_hf_export(
                hf_dir=output_dir,
                config_path=config_path,
                params=params,
            )

        print(
            "\n[export] COMPLETE"
        )

        print(
            f"[export] output dir:\n{output_dir}"
        )

    finally:
        checkpoints.close()


# ============================================================
# CLI
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint_dir",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--tokenizer_dir",
        required=True,
    )

    parser.add_argument(
        "--skip_validation",
        action="store_true",
    )

    args = parser.parse_args()

    export_hf_checkpoint(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        validate=not args.skip_validation,
    )


if __name__ == "__main__":
    main()
