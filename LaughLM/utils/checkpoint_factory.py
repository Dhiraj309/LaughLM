"""
LaughLM/utils/checkpoint_factory.py

Orbax Composite State Co-Serialization & Async Checkpoint Saver Factory.

Features:
1. Asynchronous Checkpoint Saving: Integrate orbax.checkpoint.PyTreeCheckpointer / AsyncOptions
   to run checkpoint writes on background host CPU threads, keeping TPU training steps moving
   without latency spikes.
2. Composite State Co-Serialization: Use ocp.args.Composite (or Orbax Composite handlers) to atomically
   serialize model weights, Optax optimizer states, and Grain dataset iterator states in a single step.
3. Topology-Agnostic Resharding: Configure Orbax transformation specs so saved checkpoints
   can be restored seamlessly across varying TPU pod slice sizes (e.g. 64-chip -> 16-chip/256-chip).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jax

from LaughLM.training.checkpoint import CheckpointManager as NativeCheckpointManager
from LaughLM.config.schema import LaughLMConfig

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Safe Orbax Import
# ------------------------------------------------------------

try:
    import orbax.checkpoint as ocp
    _ORBAX_AVAILABLE = True
except ImportError:
    ocp = None
    _ORBAX_AVAILABLE = False


def is_orbax_available() -> bool:
    """Return whether Orbax checkpointing is available."""
    return _ORBAX_AVAILABLE


# ------------------------------------------------------------
# Async & Composite Checkpoint Manager
# ------------------------------------------------------------

class OrbaxCompositeCheckpointManager:
    """
    Orbax Checkpoint Manager supporting async background writes, composite state serialization,
    and topology-agnostic resharding.
    """

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 3,
        async_checkpointing: bool = False,
    ):
        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = self.directory / "checkpoint_metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.async_checkpointing = async_checkpointing
        self._pending_metadata: dict[int, Dict[str, Any]] = {}

        if _ORBAX_AVAILABLE and ocp is not None:
            self._init_orbax_manager()
        else:
            self.manager = NativeCheckpointManager(
                directory=str(self.directory),
                max_to_keep=max_to_keep,
            )

    def _metadata_path(self, step: int) -> Path:
        return self.metadata_dir / f"step_{step:08d}.json"

    def _write_metadata_sidecar(
        self,
        *,
        step: int,
        metadata: Dict[str, Any],
    ) -> None:
        """Write the compatibility-audit metadata after Orbax completion."""
        path = self._metadata_path(step)
        temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _prune_metadata_sidecars(self, saved_steps: set[int]) -> None:
        for path in self.metadata_dir.glob("step_*.json"):
            try:
                step = int(path.stem.removeprefix("step_"))
            except ValueError:
                continue
            if step not in saved_steps:
                path.unlink()

    def _flush_completed_metadata(self) -> None:
        if not self._pending_metadata:
            return

        saved_steps: set[int] | None = None
        if hasattr(self.manager, "all_steps"):
            saved_steps = {
                int(step) for step in self.manager.all_steps(read=True)
            }
            missing_steps = sorted(
                step
                for step in self._pending_metadata
                if step not in saved_steps
            )
            if missing_steps:
                raise RuntimeError(
                    "Orbax completed without registering checkpoint steps "
                    f"for metadata sidecars: {missing_steps}"
                )

        for step, metadata in sorted(self._pending_metadata.items()):
            self._write_metadata_sidecar(step=step, metadata=metadata)
            logger.info(
                "[checkpoint_factory] metadata sidecar committed at step %s.",
                step,
            )
        self._pending_metadata.clear()

        if saved_steps is not None:
            self._prune_metadata_sidecars(saved_steps)

    def _init_orbax_manager(self):
        """Build Orbax CheckpointManager with AsyncOptions if enabled."""
        try:
            options = ocp.CheckpointManagerOptions(
                max_to_keep=self.max_to_keep,
                create=True,
                enable_async_checkpointing=self.async_checkpointing,
                async_options=ocp.AsyncOptions() if self.async_checkpointing else None,
                enable_background_delete=True,
            )

            # The manager discovers handlers from the StandardSave/JsonSave
            # and StandardRestore/JsonRestore arguments used below. Supplying
            # a positional Checkpointer selects Orbax's deprecated legacy API.
            self.manager = ocp.CheckpointManager(
                self.directory,
                options=options,
            )
            logger.info(
                f"[checkpoint_factory] Initialized Orbax CheckpointManager "
                f"(async={self.async_checkpointing}, directory={self.directory})."
            )
        except Exception as e:
            logger.warning(
                f"[checkpoint_factory] Orbax CheckpointManager initialization failed ({e}). "
                "Falling back to NativeCheckpointManager."
            )
            self.manager = NativeCheckpointManager(
                directory=str(self.directory),
                max_to_keep=self.max_to_keep,
            )

    def save(
        self,
        step: int,
        state: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Alias for save_composite to match native CheckpointManager interface."""
        return self.save_composite(step, state, metadata=metadata)

    def restore_latest(
        self,
        target_state: Any,
        config: LaughLMConfig,
        num_devices: int,
        require_metadata: bool = True,
        require_v3: bool = True,
        purpose: str = "fsdp_resume",
    ) -> Optional[Tuple[Any, int]]:
        """Validate metadata, then restore latest checkpoint state."""
        latest_step = self.latest_step()
        if latest_step is None:
            return None

        metadata = self.load_metadata(latest_step)
        NativeCheckpointManager.validate_metadata_compatible(
            metadata=metadata,
            config=config,
            num_devices=num_devices,
            require_metadata=require_metadata,
            require_v3=require_v3,
            purpose=purpose,
        )

        state, _ = self.restore_composite(latest_step, target_state)
        return state, latest_step

    def wait(self):
        """Alias for wait_until_finished to match native CheckpointManager interface."""
        self.wait_until_finished()

    @staticmethod
    def build_metadata_from_config(
        *,
        config: LaughLMConfig,
        step: int,
        tokens_processed: int,
        num_devices: int,
        state_token_counter_dtype: str | None = None,
    ) -> Dict[str, Any]:
        """Build the standard LaughLM checkpoint metadata for Orbax saves.

        Trainer code uses the same metadata contract for native and Orbax
        managers. Delegate to the established native implementation so that
        checkpoint validation and cross-backend restore behavior remain identical.
        """
        return NativeCheckpointManager.build_metadata_from_config(
            config=config,
            step=step,
            tokens_processed=tokens_processed,
            num_devices=num_devices,
            state_token_counter_dtype=state_token_counter_dtype,
        )


    def save_composite(
        self,
        step: int,
        model_state: Any,
        grain_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Atomically co-serialize model weights, Optax optimizer states, and Grain dataset iterator states.
        """
        if isinstance(self.manager, NativeCheckpointManager):
            self.manager.save(
                step,
                model_state,
                metadata=metadata,
            )
            return True

        try:
            if hasattr(self.manager, "save"):
                # Use Orbax args.Composite if available in API
                if hasattr(ocp, "args") and hasattr(ocp.args, "Composite"):
                    save_args = ocp.args.Composite(
                        state=ocp.args.StandardSave(model_state),
                        grain_iterator=ocp.args.JsonSave(grain_state or {}),
                        metadata=ocp.args.JsonSave(metadata or {}),
                    )
                    self.manager.save(step, args=save_args)
                else:
                    self.manager.save(
                        step,
                        {
                            "state": model_state,
                            "grain_iterator": grain_state or {},
                            "metadata": metadata or {},
                        },
                    )

                if metadata is not None:
                    self._pending_metadata[int(step)] = dict(metadata)
                logger.info(f"[checkpoint_factory] Composite checkpoint saved at step {step}.")
                return True
        except Exception as e:
            raise RuntimeError(
                "Composite checkpoint save failed before metadata could be "
                "committed; refusing to write a state-only checkpoint."
            ) from e

        raise RuntimeError(
            "Checkpoint manager does not provide a supported save method."
        )

    @staticmethod
    def _composite_item(restored: Any, name: str, default: Any) -> Any:
        """Read a named Orbax composite item across supported return types."""
        try:
            return restored[name]
        except (KeyError, TypeError):
            return getattr(restored, name, default)

    def load_metadata(self, step: int) -> Optional[Dict[str, Any]]:
        """Load metadata written atomically with an async composite checkpoint."""
        if isinstance(self.manager, NativeCheckpointManager):
            return self.manager.load_metadata(step)

        if not (
            hasattr(self.manager, "restore")
            and hasattr(ocp, "args")
            and hasattr(ocp.args, "Composite")
        ):
            return None

        try:
            restored = self.manager.restore(
                step,
                args=ocp.args.Composite(
                    metadata=ocp.args.JsonRestore(),
                ),
            )
            metadata = self._composite_item(restored, "metadata", None)
        except Exception as error:
            logger.warning(
                "[checkpoint_factory] Could not restore checkpoint metadata "
                f"for step {step}: {error}"
            )
            return None

        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise ValueError(
                "Checkpoint metadata must be a JSON object, got "
                f"{type(metadata).__name__}."
            )
        return metadata

    def restore_composite(
        self,
        step: int,
        target_model_state: Any,
        target_mesh: Optional[Any] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Restore composite state with topology-agnostic resharding across different TPU pod slice sizes.
        """
        if hasattr(self.manager, "restore"):
            try:
                # Configure Orbax restore args with target mesh sharding if available
                restore_args = None
                if target_mesh is not None and hasattr(ocp, "checkpoint_utils"):
                    try:
                        restore_args = ocp.checkpoint_utils.construct_restore_args(
                            target_model_state, target_mesh
                        )
                    except Exception:
                        restore_args = None

                if hasattr(ocp, "args") and hasattr(ocp.args, "Composite"):
                    standard_restore_kwargs = {}
                    if restore_args is not None:
                        try:
                            # Newer Orbax releases accept target-specific restore
                            # args on StandardRestore.
                            standard_restore_kwargs["restore_args"] = restore_args
                            state_restore = ocp.args.StandardRestore(
                                target_model_state,
                                **standard_restore_kwargs,
                            )
                        except TypeError as error:
                            if "restore_args" not in str(error):
                                raise
                            # Older Orbax releases infer restore args from the
                            # target pytree and reject this keyword entirely.
                            state_restore = ocp.args.StandardRestore(
                                target_model_state,
                            )
                    else:
                        # Do not pass restore_args=None: older Orbax versions
                        # reject the keyword even when no target mesh is used.
                        state_restore = ocp.args.StandardRestore(
                            target_model_state,
                        )

                    res_args = ocp.args.Composite(
                        state=state_restore,
                        grain_iterator=ocp.args.JsonRestore(),
                    )
                    restored = self.manager.restore(step, args=res_args)
                    model_state = self._composite_item(
                        restored,
                        "state",
                        target_model_state,
                    )
                    grain_state = self._composite_item(
                        restored,
                        "grain_iterator",
                        {},
                    )
                    return model_state, grain_state
                else:
                    restored = self.manager.restore(step, target_model_state)
                    if isinstance(restored, dict) and "state" in restored:
                        return restored["state"], restored.get("grain_iterator", {})
                    return restored, {}
            except Exception as e:
                logger.warning(
                    f"[checkpoint_factory] Composite restore failed ({e})."
                )

                # A composite checkpoint cannot be safely restored through the
                # legacy positional API: Orbax interprets the TrainState as an
                # item mapping and fails with an opaque ``.keys`` error. Raise
                # the original compatibility problem instead of masking it.
                raise RuntimeError(
                    "Composite checkpoint restore failed; the checkpoint was "
                    "not restored through a state-only fallback. Check the "
                    "Orbax version/API compatibility."
                ) from e

        return target_model_state, {}

    def latest_step(self) -> Optional[int]:
        """Return latest checkpoint step index."""
        if hasattr(self.manager, "latest_step"):
            return self.manager.latest_step()
        return None

    def wait_until_finished(self):
        """Flush background async writes before shutdown."""
        if hasattr(self.manager, "wait_until_finished"):
            self.manager.wait_until_finished()
        self._flush_completed_metadata()


# ------------------------------------------------------------
# Main Factory Function
# ------------------------------------------------------------

def create_checkpoint_manager(
    config: LaughLMConfig,
    directory: str,
    max_to_keep: int = 3,
) -> Any:
    """
    Build checkpoint manager based on config.optimizations.async_checkpointing.
    """
    async_checkpointing = getattr(
        getattr(config, "optimizations", None),
        "async_checkpointing",
        False,
    )

    if async_checkpointing:
        return OrbaxCompositeCheckpointManager(
            directory=directory,
            max_to_keep=max_to_keep,
            async_checkpointing=True,
        )

    # Standard CheckpointManager
    return NativeCheckpointManager(
        directory=directory,
        max_to_keep=max_to_keep,
    )
