"""
LaughLM/training/checkpoint.py

PMAP-safe Orbax checkpoint manager with resume metadata validation.

Resume policy
-------------
Strictly incompatible:
- Parameter-shape-defining model fields
- Param-tree / forward-semantics fields
- LR-schedule-defining fields after scheduler metadata exists

Allowed with warning:
- runtime.total_tokens, because this is now only the current stage stop target

Important scheduler rule
------------------------
For iterative pretraining:

    runtime.total_tokens      may increase stage-by-stage
    scheduler.horizon_tokens  must stay fixed

Changing scheduler.horizon_tokens reshapes the LR curve and is unsafe
for a clean resume.
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

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_dir = (
            self.directory
            / "checkpoint_metadata"
        )

        self.metadata_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        print(
            "[checkpoint] directory:\n"
            f"  {self.directory}"
        )

        print(
            "[checkpoint] jax process:\n"
            f"  process_index={jax.process_index()}\n"
            f"  process_count={jax.process_count()}"
        )

    # --------------------------------------------------------
    # Metadata paths
    # --------------------------------------------------------

    def _metadata_path(
        self,
        step: int,
    ) -> Path:
        return (
            self.metadata_dir
            / f"step_{step:08d}.json"
        )

    # --------------------------------------------------------
    # Metadata save/load
    # --------------------------------------------------------

    def save_metadata(
        self,
        *,
        step: int,
        metadata: dict,
    ):
        if jax.process_index() != 0:
            return

        path = self._metadata_path(
            step
        )

        with open(
            path,
            "w",
        ) as f:
            json.dump(
                metadata,
                f,
                indent=2,
                sort_keys=True,
            )

    def load_metadata(
        self,
        step: int,
    ):
        path = self._metadata_path(
            step
        )

        if not path.exists():
            return None

        with open(
            path,
            "r",
        ) as f:
            return json.load(f)

    # --------------------------------------------------------
    # Save / restore
    # --------------------------------------------------------

    def save(
        self,
        step: int,
        state,
        metadata: dict | None = None,
    ):
        if step < 0:
            raise ValueError(
                f"Invalid checkpoint step: {step}"
            )

        print(
            f"[checkpoint] saving step {step:,}"
        )

        try:
            jax.experimental.multihost_utils.sync_global_devices(
                f"checkpoint-save-{step}"
            )

            if metadata is not None:
                self.save_metadata(
                    step=step,
                    metadata=metadata,
                )

            self.manager.save(
                step,
                args=ocp.args.StandardSave(
                    state
                ),
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

    # --------------------------------------------------------
    # Metadata helpers
    # --------------------------------------------------------

    @staticmethod
    def _tokens_per_step(
        *,
        config,
        num_devices: int,
    ) -> int:
        tokens_per_step = (
            int(config.runtime.seq_len)
            * int(config.runtime.micro_batch_per_device)
            * int(num_devices)
            * int(config.runtime.gradient_accumulation)
        )

        if tokens_per_step <= 0:
            raise ValueError(
                "Computed tokens_per_step <= 0 while building checkpoint metadata."
            )

        return int(tokens_per_step)

    @staticmethod
    def _scheduler_horizon_tokens(
        config,
    ) -> int:
        horizon_tokens = getattr(
            config.scheduler,
            "horizon_tokens",
            None,
        )

        if horizon_tokens is None:
            return int(
                config.runtime.total_tokens
            )

        return int(
            horizon_tokens
        )

    @staticmethod
    def _scheduler_total_steps(
        *,
        config,
        num_devices: int,
    ) -> int:
        tokens_per_step = (
            CheckpointManager._tokens_per_step(
                config=config,
                num_devices=num_devices,
            )
        )

        horizon_tokens = (
            CheckpointManager._scheduler_horizon_tokens(
                config
            )
        )

        total_steps = (
            horizon_tokens
            // tokens_per_step
        )

        if total_steps <= 0:
            raise ValueError(
                "scheduler.horizon_tokens produces zero scheduler steps."
            )

        return int(total_steps)

    @staticmethod
    def _runtime_total_steps(
        *,
        config,
        num_devices: int,
    ) -> int:
        tokens_per_step = (
            CheckpointManager._tokens_per_step(
                config=config,
                num_devices=num_devices,
            )
        )

        total_steps = (
            int(config.runtime.total_tokens)
            // tokens_per_step
        )

        if total_steps <= 0:
            raise ValueError(
                "runtime.total_tokens produces zero runtime steps."
            )

        return int(total_steps)

    # --------------------------------------------------------
    # Build checkpoint metadata
    # --------------------------------------------------------

    @staticmethod
    def build_metadata_from_config(
        *,
        config,
        step: int,
        tokens_processed: int,
        num_devices: int,
    ) -> dict:
        arch = config.architecture

        tokens_per_step = (
            CheckpointManager._tokens_per_step(
                config=config,
                num_devices=num_devices,
            )
        )

        scheduler_horizon_tokens = (
            CheckpointManager._scheduler_horizon_tokens(
                config
            )
        )

        scheduler_total_steps = (
            CheckpointManager._scheduler_total_steps(
                config=config,
                num_devices=num_devices,
            )
        )

        runtime_total_steps = (
            CheckpointManager._runtime_total_steps(
                config=config,
                num_devices=num_devices,
            )
        )

        return {
            "format": "laughlm_pmap_checkpoint_v2",
            "step": int(step),
            "tokens_processed": int(tokens_processed),
            "num_devices": int(num_devices),
            "tokens_per_step": int(tokens_per_step),

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
                "total_tokens": int(
                    config.runtime.total_tokens
                ),
                "total_steps": int(
                    runtime_total_steps
                ),
            },

            "optimizer": {
                "type": str(config.optimizer.type),
                "learning_rate": float(
                    config.optimizer.learning_rate
                ),
                "beta1": float(config.optimizer.beta1),
                "beta2": float(config.optimizer.beta2),
                "eps": float(config.optimizer.eps),
                "weight_decay": float(
                    config.optimizer.weight_decay
                ),
                "gradient_clip": float(
                    config.optimizer.gradient_clip
                ),
                "mu_dtype": str(
                    getattr(
                        config.optimizer,
                        "mu_dtype",
                        "float32",
                    )
                ),
            },

            "scheduler": {
                "type": str(config.scheduler.type),
                "horizon_tokens": int(
                    scheduler_horizon_tokens
                ),
                "total_steps": int(
                    scheduler_total_steps
                ),
                "warmup_steps": (
                    None
                    if config.scheduler.warmup_steps is None
                    else int(config.scheduler.warmup_steps)
                ),
                "warmup_fraction": (
                    None
                    if config.scheduler.warmup_fraction is None
                    else float(config.scheduler.warmup_fraction)
                ),
                "stable_fraction": float(
                    config.scheduler.stable_fraction
                ),
                "decay_steps": (
                    None
                    if config.scheduler.decay_steps is None
                    else int(config.scheduler.decay_steps)
                ),
                "min_lr_ratio": float(
                    config.scheduler.min_lr_ratio
                ),
            },

            "parallelism": {
                "data_parallel": int(
                    config.parallelism.data_parallel
                ),
                "model_parallel": int(
                    config.parallelism.model_parallel
                ),
                "compute_dtype": str(
                    config.parallelism.compute_dtype
                ),
                "param_dtype": str(
                    config.parallelism.param_dtype
                ),
            },

            "architecture": {
                "positional": str(arch.positional),
                "normalization": str(arch.normalization),
                "attention_impl": str(arch.attention_impl),
                "attention_variant": str(arch.attention_variant),
                "parallel_block": bool(arch.parallel_block),
                "fused_qkv": bool(
                    getattr(
                        arch,
                        "fused_qkv",
                        False,
                    )
                ),
                "weight_tying": bool(
                    arch.weight_tying
                ),
            },
        }

    # --------------------------------------------------------
    # Validation helpers
    # --------------------------------------------------------

    @staticmethod
    def _compare_strict(
        *,
        name: str,
        old,
        new,
        mismatches: list[str],
    ):
        if old != new:
            mismatches.append(
                f"  {name}: checkpoint={old!r}, current={new!r}"
            )

    @staticmethod
    def _compare_flexible(
        *,
        name: str,
        old,
        new,
        changed: list[str],
    ):
        if old != new:
            changed.append(
                f"  {name}: checkpoint={old!r}, current={new!r}"
            )

    # --------------------------------------------------------
    # Validate checkpoint compatibility
    # --------------------------------------------------------

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

        meta_model = metadata.get(
            "model",
            {},
        )

        meta_arch = metadata.get(
            "architecture",
            {},
        )

        meta_runtime = metadata.get(
            "runtime",
            {},
        )

        meta_parallel = metadata.get(
            "parallelism",
            {},
        )

        meta_optimizer = metadata.get(
            "optimizer",
            None,
        )

        meta_scheduler = metadata.get(
            "scheduler",
            None,
        )

        current_kv_heads = (
            config.model.num_kv_heads
        )

        # ----------------------------------------------------
        # Strict model / parameter-tree checks
        # ----------------------------------------------------

        mismatches = []

        strict_checks = {
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
                getattr(
                    config.architecture,
                    "fused_qkv",
                    False,
                ),
            ),
            "architecture.weight_tying": (
                meta_arch.get("weight_tying"),
                config.architecture.weight_tying,
            ),
        }

        for name, (old, new) in strict_checks.items():
            CheckpointManager._compare_strict(
                name=name,
                old=old,
                new=new,
                mismatches=mismatches,
            )

        old_positional = meta_arch.get(
            "positional"
        )

        new_positional = (
            config.architecture.positional
        )

        if (
            old_positional == "learned"
            or new_positional == "learned"
        ):
            old_max_seq = meta_model.get(
                "max_seq_len"
            )

            new_max_seq = (
                config.model.max_seq_len
            )

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

        # ----------------------------------------------------
        # Strict optimizer / scheduler checks
        # ----------------------------------------------------

        if (
            meta_optimizer is None
            or meta_scheduler is None
        ):
            print(
                "[checkpoint] warning: old checkpoint metadata has no "
                "optimizer/scheduler block; cannot strictly validate LR resume safety",
                flush=True,
            )

        else:
            lr_mismatches = []

            current_scheduler_horizon = (
                CheckpointManager._scheduler_horizon_tokens(
                    config
                )
            )

            current_scheduler_total_steps = (
                CheckpointManager._scheduler_total_steps(
                    config=config,
                    num_devices=num_devices,
                )
            )

            strict_lr_checks = {
                "optimizer.type": (
                    meta_optimizer.get("type"),
                    str(config.optimizer.type),
                ),
                "optimizer.learning_rate": (
                    meta_optimizer.get("learning_rate"),
                    float(config.optimizer.learning_rate),
                ),
                "scheduler.type": (
                    meta_scheduler.get("type"),
                    str(config.scheduler.type),
                ),
                "scheduler.horizon_tokens": (
                    meta_scheduler.get("horizon_tokens"),
                    int(current_scheduler_horizon),
                ),
                "scheduler.total_steps": (
                    meta_scheduler.get("total_steps"),
                    int(current_scheduler_total_steps),
                ),
                "scheduler.warmup_steps": (
                    meta_scheduler.get("warmup_steps"),
                    (
                        None
                        if config.scheduler.warmup_steps is None
                        else int(config.scheduler.warmup_steps)
                    ),
                ),
                "scheduler.warmup_fraction": (
                    meta_scheduler.get("warmup_fraction"),
                    (
                        None
                        if config.scheduler.warmup_fraction is None
                        else float(config.scheduler.warmup_fraction)
                    ),
                ),
                "scheduler.stable_fraction": (
                    meta_scheduler.get("stable_fraction"),
                    float(config.scheduler.stable_fraction),
                ),
                "scheduler.decay_steps": (
                    meta_scheduler.get("decay_steps"),
                    (
                        None
                        if config.scheduler.decay_steps is None
                        else int(config.scheduler.decay_steps)
                    ),
                ),
                "scheduler.min_lr_ratio": (
                    meta_scheduler.get("min_lr_ratio"),
                    float(config.scheduler.min_lr_ratio),
                ),
            }

            for name, (old, new) in strict_lr_checks.items():
                CheckpointManager._compare_strict(
                    name=name,
                    old=old,
                    new=new,
                    mismatches=lr_mismatches,
                )

            if lr_mismatches:
                raise ValueError(
                    "Checkpoint LR schedule is not compatible with current config.\n"
                    "This would reshape or restart the LR curve.\n"
                    "For iterative training, change runtime.total_tokens only and keep "
                    "scheduler.horizon_tokens plus scheduler fields fixed.\n"
                    + "\n".join(lr_mismatches)
                )

        # ----------------------------------------------------
        # Flexible runtime/stage checks
        # ----------------------------------------------------

        changed = []

        flexible_checks = {
            "runtime.total_tokens": (
                meta_runtime.get("total_tokens"),
                config.runtime.total_tokens,
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

        for name, (old, new) in flexible_checks.items():
            CheckpointManager._compare_flexible(
                name=name,
                old=old,
                new=new,
                changed=changed,
            )

        if changed:
            print(
                "[checkpoint] resume with changed stage/runtime config:\n"
                + "\n".join(changed),
                flush=True,
            )

        # ----------------------------------------------------
        # Runtime shape fields that affect tokens_per_step
        # ----------------------------------------------------
        # These are intentionally strict because the LR schedule is
        # step-based. Changing tokens_per_step after resume changes
        # LR-vs-token semantics.

        old_tokens_per_step = metadata.get(
            "tokens_per_step"
        )

        new_tokens_per_step = (
            CheckpointManager._tokens_per_step(
                config=config,
                num_devices=num_devices,
            )
        )

        if (
            old_tokens_per_step is not None
            and int(old_tokens_per_step) != int(new_tokens_per_step)
        ):
            raise ValueError(
                "tokens_per_step changed across resume.\n"
                "This changes optimizer update frequency and LR-vs-token semantics.\n"
                "Use a fresh checkpoint_dir for batch/sequence/GA experiments.\n"
                f"  checkpoint tokens_per_step: {int(old_tokens_per_step):,}\n"
                f"  current tokens_per_step:    {int(new_tokens_per_step):,}"
            )

        runtime_shape_checks = {
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
        }

        runtime_shape_changes = []

        for name, (old, new) in runtime_shape_checks.items():
            CheckpointManager._compare_flexible(
                name=name,
                old=old,
                new=new,
                changed=runtime_shape_changes,
            )

        if runtime_shape_changes:
            print(
                "[checkpoint] runtime shape changed but tokens_per_step matched:\n"
                + "\n".join(runtime_shape_changes),
                flush=True,
            )

    # --------------------------------------------------------
    # Restore
    # --------------------------------------------------------

    def restore_latest(
        self,
        target_state,
        *,
        config=None,
        num_devices: int | None = None,
    ):
        latest = self.manager.latest_step()

        if latest is None:
            print(
                "[checkpoint] no checkpoint found",
                flush=True,
            )

            return None

        print(
            f"[checkpoint] restoring step {latest}",
            flush=True,
        )

        metadata = self.load_metadata(
            latest
        )

        if metadata is not None:
            print(
                "[checkpoint] metadata:\n"
                f"  step={metadata.get('step')}\n"
                f"  tokens_processed={metadata.get('tokens_processed')}\n"
                f"  format={metadata.get('format')}"
            )

        if (
            config is not None
            and num_devices is not None
        ):
            self.validate_metadata_compatible(
                metadata=metadata,
                config=config,
                num_devices=num_devices,
            )

        try:
            restored = self.manager.restore(
                latest,
                args=ocp.args.StandardRestore(
                    target_state
                ),
            )

            print(
                f"[checkpoint] restored step {latest}",
                flush=True,
            )

            return restored, latest

        except Exception as e:
            print(
                "[checkpoint] RESTORE FAILED",
                flush=True,
            )

            print(
                type(e).__name__ + ":",
                str(e),
                flush=True,
            )

            traceback.print_exc()

            raise

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    def close(self):
        try:
            self.wait()

            jax.experimental.multihost_utils.sync_global_devices(
                "checkpoint-close"
            )

        finally:
            self.manager.close()
