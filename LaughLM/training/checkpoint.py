"""
LaughLM/training/checkpoint.py

PMAP-safe Orbax checkpoint manager with resume metadata validation.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import jax
import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(
        self,
        directory: str,
        max_to_keep: int = 3,
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.directory / "checkpoint_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            enable_async_checkpointing=True,
            async_options=ocp.AsyncOptions(),
            enable_background_delete=True,
        )

        checkpointer = ocp.Checkpointer(
            ocp.StandardCheckpointHandler()
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            checkpointer,
            options=options,
        )

        print("[checkpoint] directory:\n" f"  {self.directory}")
        print(
            "[checkpoint] jax process:\n"
            f"  process_index={jax.process_index()}\n"
            f"  process_count={jax.process_count()}"
        )

    def _metadata_path(self, step: int) -> Path:
        return self.metadata_dir / f"step_{step:08d}.json"

    def save_metadata(self, *, step: int, metadata: dict):
        if jax.process_index() != 0:
            return

        path = self._metadata_path(step)

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def load_metadata(self, step: int):
        path = self._metadata_path(step)

        if not path.exists():
            return None

        with open(path, "r") as f:
            return json.load(f)

    def save(
        self,
        step: int,
        state,
        metadata: dict | None = None,
    ):
        if step < 0:
            raise ValueError(f"Invalid checkpoint step: {step}")

        print(f"[checkpoint] saving step {step:,}")

        try:
            jax.experimental.multihost_utils.sync_global_devices(
                f"checkpoint-save-{step}"
            )

            if metadata is not None:
                self.save_metadata(step=step, metadata=metadata)

            self.manager.save(
                step,
                args=ocp.args.StandardSave(state),
            )

        except Exception as e:
            print(
                "[checkpoint] SAVE FAILED\n"
                f"{type(e).__name__}: {e}"
            )
            traceback.print_exc()
            raise

    def wait(self):
        self.manager.wait_until_finished()

    def latest_step(self):
        return self.manager.latest_step()

    @staticmethod
    def build_metadata_from_config(
        *,
        config,
        step: int,
        tokens_processed: int,
        num_devices: int,
    ) -> dict:
        return {
            "format": "laughlm_pmap_checkpoint_v1",
            "step": int(step),
            "tokens_processed": int(tokens_processed),
            "num_devices": int(num_devices),
            "model": {
                "vocab_size": int(config.model.vocab_size),
                "d_model": int(config.model.d_model),
                "num_layers": int(config.model.num_layers),
                "num_heads": int(config.model.num_heads),
                "num_kv_heads": (
                    None
                    if config.model.num_kv_heads is None
                    else int(config.model.num_kv_heads)
                ),
                "max_seq_len": int(config.model.max_seq_len),
            },
            "runtime": {
                "seq_len": int(config.runtime.seq_len),
                "micro_batch_per_device": int(
                    config.runtime.micro_batch_per_device
                ),
                "gradient_accumulation": int(
                    config.runtime.gradient_accumulation
                ),
            },
            "parallelism": {
                "data_parallel": int(config.parallelism.data_parallel),
                "model_parallel": int(config.parallelism.model_parallel),
                "compute_dtype": str(config.parallelism.compute_dtype),
                "param_dtype": str(config.parallelism.param_dtype),
            },
            "architecture": {
                "attention_impl": str(config.architecture.attention_impl),
                "attention_variant": str(config.architecture.attention_variant),
                "parallel_block": bool(config.architecture.parallel_block),
                "weight_tying": bool(config.architecture.weight_tying),
            },
        }

    @staticmethod
    def validate_metadata_compatible(
        *,
        metadata: dict | None,
        config,
        num_devices: int,
    ):
        if metadata is None:
            print(
                "[checkpoint] warning: no metadata found; "
                "cannot validate resume compatibility",
                flush=True,
            )
            return

        checks = {
            "model.vocab_size": (
                metadata["model"]["vocab_size"],
                config.model.vocab_size,
            ),
            "model.d_model": (
                metadata["model"]["d_model"],
                config.model.d_model,
            ),
            "model.num_layers": (
                metadata["model"]["num_layers"],
                config.model.num_layers,
            ),
            "model.num_heads": (
                metadata["model"]["num_heads"],
                config.model.num_heads,
            ),
            "model.num_kv_heads": (
                metadata["model"]["num_kv_heads"],
                config.model.num_kv_heads,
            ),
            "model.max_seq_len": (
                metadata["model"]["max_seq_len"],
                config.model.max_seq_len,
            ),
            "runtime.seq_len": (
                metadata["runtime"]["seq_len"],
                config.runtime.seq_len,
            ),
            "runtime.micro_batch_per_device": (
                metadata["runtime"]["micro_batch_per_device"],
                config.runtime.micro_batch_per_device,
            ),
            "runtime.gradient_accumulation": (
                metadata["runtime"]["gradient_accumulation"],
                config.runtime.gradient_accumulation,
            ),
            "parallelism.data_parallel": (
                metadata["parallelism"]["data_parallel"],
                config.parallelism.data_parallel,
            ),
            "parallelism.compute_dtype": (
                metadata["parallelism"]["compute_dtype"],
                config.parallelism.compute_dtype,
            ),
            "parallelism.param_dtype": (
                metadata["parallelism"]["param_dtype"],
                config.parallelism.param_dtype,
            ),
            "num_devices": (
                metadata["num_devices"],
                num_devices,
            ),
        }

        mismatches = []

        for name, (old, new) in checks.items():
            if old != new:
                mismatches.append(
                    f"  {name}: checkpoint={old!r}, current={new!r}"
                )

        if mismatches:
            raise ValueError(
                "Checkpoint is not compatible with current config.\n"
                "Use a fresh checkpoint_dir or restore with matching config.\n"
                + "\n".join(mismatches)
            )

    def restore_latest(
        self,
        target_state,
        *,
        config=None,
        num_devices: int | None = None,
    ):
        latest = self.manager.latest_step()

        if latest is None:
            print("[checkpoint] no checkpoint found", flush=True)
            return None

        print(f"[checkpoint] restoring step {latest}", flush=True)

        metadata = self.load_metadata(latest)

        if metadata is not None:
            print(
                "[checkpoint] metadata:\n"
                f"  step={metadata.get('step')}\n"
                f"  tokens_processed={metadata.get('tokens_processed')}\n"
                f"  format={metadata.get('format')}"
            )

        if config is not None and num_devices is not None:
            self.validate_metadata_compatible(
                metadata=metadata,
                config=config,
                num_devices=num_devices,
            )

        try:
            restored = self.manager.restore(
                latest,
                args=ocp.args.StandardRestore(target_state),
            )

            print(f"[checkpoint] restored step {latest}", flush=True)

            return restored, latest

        except Exception as e:
            print("[checkpoint] RESTORE FAILED", flush=True)
            print(type(e).__name__ + ":", str(e), flush=True)
            traceback.print_exc()
            raise

    def close(self):
        try:
            self.wait()
            jax.experimental.multihost_utils.sync_global_devices(
                "checkpoint-close"
            )
        finally:
            self.manager.close()