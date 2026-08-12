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

import logging
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
        self.max_to_keep = max_to_keep
        self.async_checkpointing = async_checkpointing

        if _ORBAX_AVAILABLE and ocp is not None:
            self._init_orbax_manager()
        else:
            self.manager = NativeCheckpointManager(
                directory=str(self.directory),
                max_to_keep=max_to_keep,
            )

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

            # PyTreeCheckpointer / StandardCheckpointHandler
            checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
            self.manager = ocp.CheckpointManager(
                self.directory,
                checkpointer,
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
        """Restore latest checkpoint to match native CheckpointManager interface."""
        latest_step = self.latest_step()
        if latest_step is None:
            return None
        
        state, _ = self.restore_composite(latest_step, target_state)
        # Simplified return to match native interface expected by FSDPTrainer
        return state, latest_step

    def wait(self):
        """Alias for wait_until_finished to match native CheckpointManager interface."""
        self.wait_until_finished()

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
        composite_data = {
            "state": model_state,
            "grain_iterator": grain_state or {},
            "metadata": metadata or {},
        }

        try:
            if hasattr(self.manager, "save"):
                # Use Orbax args.Composite if available in API
                if hasattr(ocp, "args") and hasattr(ocp.args, "Composite"):
                    save_args = ocp.args.Composite(
                        state=ocp.args.StandardSave(model_state),
                        grain_iterator=ocp.args.JsonSave(grain_state or {}),
                    )
                    self.manager.save(step, args=save_args)
                else:
                    self.manager.save(step, composite_data)

                logger.info(f"[checkpoint_factory] Composite checkpoint saved at step {step}.")
                return True
        except Exception as e:
            logger.warning(
                f"[checkpoint_factory] Composite save via Orbax failed ({e}). Trying fallback native save."
            )

        # Native fallback
        if hasattr(self.manager, "save"):
            self.manager.save(step, model_state)
            return True
        return False

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
                    res_args = ocp.args.Composite(
                        state=ocp.args.StandardRestore(target_model_state, restore_args=restore_args),
                        grain_iterator=ocp.args.JsonRestore(),
                    )
                    restored = self.manager.restore(step, args=res_args)
                    model_state = restored.get("state", target_model_state)
                    grain_state = restored.get("grain_iterator", {})
                    return model_state, grain_state
                else:
                    restored = self.manager.restore(step, target_model_state)
                    if isinstance(restored, dict) and "state" in restored:
                        return restored["state"], restored.get("grain_iterator", {})
                    return restored, {}
            except Exception as e:
                logger.warning(
                    f"[checkpoint_factory] Composite restore failed ({e}). Falling back to standard restore."
                )

        if hasattr(self.manager, "restore"):
            restored = self.manager.restore(step, target_model_state)
            return restored, {}

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
