"""
scripts/profile_training.py

Benchmark CLI script for running LaughLM performance profiling.

Usage:
    python scripts/profile_training.py \
        --config configs/v5e_smoke.yaml \
        --steps 100 \
        --warmup 20 \
        --level detailed \
        --output_dir profiles
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
import numpy as np

# Ensure LaughLM root is on Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from LaughLM.config.loader import load_config
from LaughLM.profiling.core.profiler import Profiler


def create_dummy_dataloader(global_batch_size: int, seq_len: int):
    """
    Generator creating synthetic token batches for isolated benchmarking when disk dataset is missing.
    """
    rng = np.random.default_rng(seed=42)
    while True:
        yield rng.integers(0, 1000, size=(global_batch_size, seq_len), dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="LaughLM Performance Profiler Benchmark CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v5e_smoke.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Total steps to profile",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of initial warmup steps to ignore before profiling",
    )
    parser.add_argument(
        "--level",
        type=str,
        choices=["off", "summary", "detailed", "developer"],
        default="detailed",
        help="Profiling detail level",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="profiles",
        help="Root directory for profile artifacts",
    )
    parser.add_argument(
        "--xprof",
        action="store_true",
        help="Enable JAX/XProf low-level tracing",
    )
    parser.add_argument(
        "--layer_profiling",
        action="store_true",
        help="Enable individual transformer layer profiling",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Custom run ID for artifact naming",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[profile_training] Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"[profile_training] Loading config: {config_path}", flush=True)
    config = load_config(config_path)

    # Isolated safety: override checkpoint directory to temporary path inside profiles
    safe_ckpt_dir = tempfile.mkdtemp(prefix="profile_ckpt_")
    config.runtime.checkpoint_dir = safe_ckpt_dir

    # Configure profiling
    config.profiling.enabled = True
    config.profiling.level = args.level
    config.profiling.output_dir = args.output_dir
    config.profiling.xprof = args.xprof
    config.profiling.layer_profiling = args.layer_profiling
    config.profiling.warmup_steps = args.warmup
    config.profiling.active_steps = args.steps

    # Instantiate profiler
    profiler = Profiler.from_config(config, run_id=args.run_id)

    print(
        f"[profile_training] Starting profile run [{profiler.run_id}] "
        f"level={args.level} steps={args.steps} warmup={args.warmup}",
        flush=True,
    )

    # Determine backend
    backend = getattr(config.runtime, "canonical_backend", config.runtime.backend)

    # Global batch size
    if backend == "fsdp":
        global_batch_size = config.runtime.micro_batch_per_device * getattr(config.parallelism, "data_parallel", 1)
    else:
        global_batch_size = config.runtime.micro_batch_per_device * getattr(config.parallelism, "data_parallel", 1)

    dataloader = create_dummy_dataloader(
        global_batch_size=global_batch_size,
        seq_len=config.runtime.seq_len,
    )

    if backend == "fsdp":
        from LaughLM.training.fsdp_trainer import FSDPTrainer
        trainer = FSDPTrainer(config, profiler=profiler)
    else:
        from LaughLM.training.trainer import Trainer
        trainer = Trainer(config, profiler=profiler)

    # Run training loop with profiler attached
    trainer.train(dataloader)

    print(f"[profile_training] Benchmark profiling complete. Output directory: {profiler.session.output_dir}", flush=True)


if __name__ == "__main__":
    main()
