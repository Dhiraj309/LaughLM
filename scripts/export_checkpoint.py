"""
scripts/export_checkpoint.py

Export a LaughLM training checkpoint to a standalone native params file.

Important
---------
This is NOT Hugging Face export.

This script exports canonical/native LaughLM params from a TrainState
checkpoint into:

    output_dir/
        params.msgpack
        config.json
        metadata.json
        source_checkpoint_metadata.json

Safety rules
------------
- Uses current LLaMA model structure, not legacy GPTModel.
- Requires v3 checkpoint metadata.
- Validates backend/layout/dtype/model/scheduler compatibility before restore.
- Refuses missing or legacy metadata for export.

Usage
-----
python -m scripts.export_checkpoint \
    --checkpoint_dir checkpoints \
    --output_dir exported_model \
    --config configs/v5e_pmap.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax.serialization import to_bytes

from LaughLM.config.loader import load_config
from LaughLM.model.llama.config_factory import build_llama_config
from LaughLM.model.llama.model import LlamaForCausalLM
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.optimizer import build_optimizer
from LaughLM.training.scheduler import build_scheduler
from LaughLM.training.train_state import TrainState
from LaughLM.utils.rng import create_rng


def _canonical_backend(config) -> str:
    return str(
        getattr(
            config.runtime,
            "canonical_backend",
            config.runtime.backend,
        )
    )


def _metadata_num_devices(config) -> int:
    """
    Return the num_devices value used for checkpoint metadata validation.

    PMAP metadata uses local replicated device count.

    FSDP metadata intentionally uses the active data replicas count because
    tokens_per_step is:
        seq_len * micro_batch_per_device * data_replicas * grad_accum
    """

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
        "Native checkpoint export only supports current production backends.\n"
        f"  runtime.backend={config.runtime.backend!r}\n"
        f"  canonical_backend={backend!r}\n"
        "Reserved backends such as parallel3d/moe need an explicit "
        "canonical unshard/export path before export is allowed."
    )


def _to_jsonable(value: Any):
    try:
        value = jax.device_get(value)
    except Exception:
        pass

    try:
        return int(value)
    except Exception:
        pass

    try:
        return float(value)
    except Exception:
        pass

    return value


def _build_target_state(config, *, num_devices: int) -> TrainState:
    """
    Build target TrainState structure for Orbax restore.

    The target structure must match the current LLaMA params/optimizer state.
    Backend-specific placement/sharding is intentionally not handled here.
    """

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
        # For tied embeddings, avoid one-time full logits materialization.
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export a LaughLM checkpoint to native params.msgpack. "
            "This is not Hugging Face export."
        )
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing Orbax checkpoints.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="exported_model",
        help="Output directory for native exported params.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config YAML used for training.",
    )

    args = parser.parse_args()

    if jax.process_count() != 1:
        raise RuntimeError(
            "scripts/export_checkpoint.py currently supports single-host "
            "exports only.\n"
            "Multi-host/FSDP export requires the Phase 4B canonical "
            "unshard/gather path before it is safe."
        )

    print(
        f"[export] Loading config: {args.config}"
    )

    config = load_config(
        args.config
    )

    backend = _canonical_backend(
        config
    )

    num_devices = _metadata_num_devices(
        config
    )

    print(
        "[export] backend:\n"
        f"  raw={config.runtime.backend}\n"
        f"  canonical={backend}\n"
        f"  metadata_num_devices={num_devices}"
    )

    print(
        "[export] Initializing LLaMA target TrainState..."
    )

    target_state = _build_target_state(
        config,
        num_devices=num_devices,
    )

    print(
        f"[export] Loading checkpoint from: {args.checkpoint_dir}"
    )

    ckpt_manager = CheckpointManager(
        args.checkpoint_dir,
        max_to_keep=99,
    )

    result = ckpt_manager.restore_latest(
        target_state=target_state,
        config=config,
        num_devices=num_devices,
        require_metadata=True,
        require_v3=True,
        purpose="native_export",
    )

    if result is None:
        raise FileNotFoundError(
            f"No checkpoint found in {args.checkpoint_dir!r}"
        )

    state, step = result

    source_metadata = ckpt_manager.load_metadata(
        step
    )

    tokens_processed = _to_jsonable(
        state.tokens_processed
    )

    print(
        "[export] restored:\n"
        f"  step={int(step):,}\n"
        f"  tokens_processed={tokens_processed:,}"
    )

    params = jax.device_get(
        state.params
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    params_path = (
        output_dir
        / "params.msgpack"
    )

    print(
        f"[export] Saving native params to: {params_path}"
    )

    params_bytes = to_bytes(
        params
    )

    with open(
        params_path,
        "wb",
    ) as f:
        f.write(
            params_bytes
        )

    print(
        f"[export] Params size: {len(params_bytes) / 1e6:.1f} MB"
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
            config.model_dump(),
            f,
            indent=2,
        )

    print(
        f"[export] Config saved to: {config_path}"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    export_metadata = {
        "format": "laughlm_native_params_export_v1",
        "step": int(step),
        "tokens_processed": int(tokens_processed),
        "backend": backend,
        "raw_backend": str(config.runtime.backend),
        "source_checkpoint": str(args.checkpoint_dir),
        "config_file": str(args.config),
        "params_file": "params.msgpack",
        "is_huggingface_export": False,
        "requires_hf_conversion": True,
    }

    with open(
        metadata_path,
        "w",
    ) as f:
        json.dump(
            export_metadata,
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        f"[export] Metadata saved to: {metadata_path}"
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
        "[export] Source checkpoint metadata saved to: "
        f"{source_metadata_path}"
    )

    ckpt_manager.close()

    print(
        f"\n[export] ✅ Native export complete → {output_dir}/"
    )
    print(
        "[export] NOTE: This is not Hugging Face export. "
        "Run the Phase 4B HF export path after canonical conversion/parity "
        "validation is implemented."
    )


if __name__ == "__main__":
    main()
