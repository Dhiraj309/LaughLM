"""
scripts/export_checkpoint.py

Export a training checkpoint to a standalone model file.

Extracts params from the Orbax TrainState checkpoint and saves them
as a lightweight msgpack file that can be loaded without optimizer state.

Usage:
    python -m scripts.export_checkpoint \
        --checkpoint_dir checkpoints \
        --output_dir exported_model \
        --config configs/tpu_v5e_8.yaml

This produces:
    exported_model/
        params.msgpack    (model weights only, ~370MB for 184M params)
        config.json       (full experiment config for reproducibility)
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes

from LaughLM.config.loader import load_config
from LaughLM.model.gpt import GPTModel
from LaughLM.training.checkpoint import CheckpointManager
from LaughLM.training.train_state import TrainState
from LaughLM.utils.rng import create_rng


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint to standalone model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing Orbax checkpoints")
    parser.add_argument("--output_dir", type=str, default="exported_model",
                        help="Output directory for exported model")
    parser.add_argument("--config", type=str, default="configs/tpu_v5e_8.yaml",
                        help="Config YAML used for training")
    args = parser.parse_args()

    print(f"[export] Loading config: {args.config}")
    config = load_config(args.config)

    print(f"[export] Initializing model for param structure...")
    model = GPTModel(config=config)
    rng = create_rng(seed=0)

    # Create dummy input to get param structure
    dummy = jnp.zeros(
        (1, config.runtime.seq_len),
        dtype=jnp.int32,
    )
    params = model.init(rng.next_key(), dummy)["params"]

    # Build a target TrainState for restore (Orbax needs the structure)
    from LaughLM.training.optimizer import build_optimizer
    from LaughLM.training.scheduler import build_scheduler

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

    # Restore checkpoint
    print(f"[export] Loading checkpoint from: {args.checkpoint_dir}")
    ckpt_manager = CheckpointManager(args.checkpoint_dir, max_to_keep=99)
    result = ckpt_manager.restore_latest(target_state=target_state)

    if result is None:
        print("[export] ERROR: No checkpoint found!")
        return

    state, step = result
    print(f"[export] Restored checkpoint at step {step}")
    print(f"[export] Tokens processed: {state.tokens_processed:,}")

    # Extract params only
    params = state.params

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params as msgpack
    params_path = output_dir / "params.msgpack"
    print(f"[export] Saving params to: {params_path}")
    params_bytes = to_bytes(params)
    with open(params_path, "wb") as f:
        f.write(params_bytes)
    print(f"[export] Params size: {len(params_bytes) / 1e6:.1f} MB")

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)
    print(f"[export] Config saved to: {config_path}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump({
            "step": int(step),
            "tokens_processed": int(state.tokens_processed),
            "source_checkpoint": str(args.checkpoint_dir),
            "config_file": str(args.config),
        }, f, indent=2)
    print(f"[export] Metadata saved to: {meta_path}")

    print(f"\n[export] ✅ Export complete → {output_dir}/")


if __name__ == "__main__":
    main()
