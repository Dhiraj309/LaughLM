"""
LaughLM/training/checkpoint.py

PMAP-safe Orbax checkpoint manager with resume metadata validation.

Resume policy
-------------
Strictly incompatible:
- Parameter-shape-defining model fields
- Param-tree/semantics fields such as fused_qkv, parallel_block, weight_tying

Allowed with warning:
- runtime.seq_len
- runtime.micro_batch_per_device
- runtime.gradient_accumulation
- data_parallel / num_devices
- max_seq_len for RoPE/ALiBi/non-learned positional embeddings

This allows practical continuation such as:
    seq_len 2048, batch 16
    -> seq_len 8192, batch 4

as long as the parameter tree is compatible.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

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
        arch = config.architecture

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
                "positional": str(arch.positional),
                "normalization": str(arch.normalization),
                "attention_impl": str(arch.attention_impl),
                "attention_variant": str(arch.attention_variant),
                "parallel_block": bool(arch.parallel_block),
                "fused_qkv": bool(getattr(arch, "fused_qkv", False)),
                "weight_tying": bool(arch.weight_tying),
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

        meta_model = metadata.get("model", {})
        meta_arch = metadata.get("architecture", {})
        meta_runtime = metadata.get("runtime", {})
        meta_parallel = metadata.get("parallelism", {})

        current_kv_heads = config.model.num_kv_heads

        strict_checks = {
            # Parameter-shape-defining fields.
            "model.vocab_size": (
                meta_model.get("vocab_size"),
                config.model.vocab_size,
            ),
            "model.d_model": (
                meta_model.get("d_model"),
                config.model.d_model,
            ),
            "model.num_layers": (
                meta_model.get("num_layers"),
                config.model.num_layers,
            ),
            "model.num_heads": (
                meta_model.get("num_heads"),
                config.model.num_heads,
            ),
            "model.num_kv_heads": (
                meta_model.get("num_kv_heads"),
                current_kv_heads,
            ),

            # Param-tree / forward-semantics fields.
            "architecture.attention_variant": (
                meta_arch.get("attention_variant"),
                config.architecture.attention_variant,
            ),
            "architecture.parallel_block": (
                meta_arch.get("parallel_block"),
                config.architecture.parallel_block,
            ),
            "architecture.fused_qkv": (
                meta_arch.get("fused_qkv", False),
                getattr(config.architecture, "fused_qkv", False),
            ),
            "architecture.weight_tying": (
                meta_arch.get("weight_tying"),
                config.architecture.weight_tying,
            ),
        }

        mismatches = []

        for name, (old, new) in strict_checks.items():
            if old != new:
                mismatches.append(
                    f"  {name}: checkpoint={old!r}, current={new!r}"
                )

        old_positional = meta_arch.get("positional")
        new_positional = config.architecture.positional

        if old_positional == "learned" or new_positional == "learned":
            old_max_seq = meta_model.get("max_seq_len")
            new_max_seq = config.model.max_seq_len

            if old_max_seq != new_max_seq:
                mismatches.append(
                    "  model.max_seq_len: "
                    f"checkpoint={old_max_seq!r}, current={new_max_seq!r} "
                    "(strict because learned positional embeddings are parameter-shaped)"
                )

        if mismatches:
            raise ValueError(
                "Checkpoint is not compatible with current model/parameter config.\n"
                "Use a fresh checkpoint_dir or restore with matching model config.\n"
                + "\n".join(mismatches)
            )

        # Runtime-shape fields are intentionally flexible.
        flexible_checks = {
            "runtime.seq_len": (
                meta_runtime.get("seq_len"),
                config.runtime.seq_len,
            ),
            "runtime.micro_batch_per_device": (
                meta_runtime.get("micro_batch_per_device"),
                config.runtime.micro_batch_per_device,
            ),
            "runtime.gradient_accumulation": (
                meta_runtime.get("gradient_accumulation"),
                config.runtime.gradient_accumulation,
            ),
            "model.max_seq_len": (
                meta_model.get("max_seq_len"),
                config.model.max_seq_len,
            ),
            "parallelism.data_parallel": (
                meta_parallel.get("data_parallel"),
                config.parallelism.data_parallel,
            ),
            "num_devices": (
                metadata.get("num_devices"),
                num_devices,
            ),
        }

        changed = []

        for name, (old, new) in flexible_checks.items():
            if old != new:
                changed.append(
                    f"  {name}: checkpoint={old!r}, current={new!r}"
                )

        if changed:
            print(
                "[checkpoint] resume with changed runtime shape:\n"
                + "\n".join(changed),
                flush=True,
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