"""
LaughLM/training/checkpoint.py

Frontier-grade async checkpoint manager for LaughLM.

Frontier-grade additions
──────────────────────────────────────────────
1. Async Orbax checkpoint saves
2. Sharded restore support
3. NamedSharding-aware restore
4. Zero full-state host gathers
5. Mesh-native restore semantics
6. Corruption-resistant restore flow
7. Resume-safe training support
8. Abstract-state restore compatibility

References
──────────────────────────────────────────────
- MaxText
- T5X
- Orbax
"""

from pathlib import Path
import traceback

import orbax.checkpoint as ocp


class CheckpointManager:
    """
    Async-capable Orbax checkpoint manager.

    Notes
    -----
    - Saves are asynchronous (non-blocking)
    - Restore waits for pending writes automatically
    - Supports sharded TrainState restore
    - Avoids full parameter materialization
    - Preserves GSPMD shardings
    """

    def __init__(
        self,
        directory: str,
        max_to_keep: int = 3,
    ):

        self.directory = (
            Path(directory)
            .expanduser()
            .resolve()
        )

        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------
        # Orbax options
        # --------------------------------------------------

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            enable_async_checkpointing=True,
            async_options=ocp.AsyncOptions(),
        )

        # --------------------------------------------------
        # Checkpointer
        # --------------------------------------------------

        checkpointer = (
            ocp.PyTreeCheckpointer()
        )

        # --------------------------------------------------
        # Manager
        # --------------------------------------------------

        self.manager = ocp.CheckpointManager(
            self.directory,
            checkpointer,
            options=options,
        )

        print(
            f"[checkpoint] directory: "
            f"{self.directory}"
        )

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────

    def save(
        self,
        step: int,
        state,
    ):
        """
        Async save.

        IMPORTANT
        ─────────────────────────────────────
        State remains sharded.

        No device_get().
        No host materialization.
        """

        if step < 0:

            raise ValueError(
                f"Invalid checkpoint step: {step}"
            )

        print(
            f"[checkpoint] saving "
            f"step {step:,}"
        )

        try:

            save_args = (
                ocp.args.StandardSave(
                    state
                )
            )

            # ------------------------------------------
            # Async save
            # ------------------------------------------

            self.manager.save(
                step,
                args=save_args,
            )

        except Exception as e:

            print(
                "[checkpoint] ERROR during save:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            raise

    # ─────────────────────────────────────────────
    # Wait for async completion
    # ─────────────────────────────────────────────

    def wait(self):
        """
        Block until pending writes finish.
        """

        try:

            self.manager.wait_until_finished()

            print(
                "[checkpoint] "
                "all pending writes complete"
            )

        except Exception as e:

            print(
                "[checkpoint] ERROR waiting "
                "for async save:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            raise

    # ─────────────────────────────────────────────
    # Latest checkpoint helper
    # ─────────────────────────────────────────────

    def latest_step(self):

        try:

            return self.manager.latest_step()

        except Exception as e:

            print(
                "[checkpoint] ERROR reading "
                "latest step:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            return None

    # ─────────────────────────────────────────────
    # Restore latest checkpoint
    # ─────────────────────────────────────────────

    def restore_latest(
        self,
        target_state=None,
    ):
        """
        Restore latest checkpoint.

        Parameters
        ----------
        target_state:
            Abstract or initialized sharded
            TrainState used to define:
            - structure
            - dtypes
            - shardings

        IMPORTANT
        ─────────────────────────────────────
        This enables:
        - mesh-native restore
        - sharded restore
        - no full gathers
        """

        # --------------------------------------------------
        # Ensure async writes complete
        # --------------------------------------------------

        self.wait()

        latest_step = self.latest_step()

        if latest_step is None:

            print(
                "[checkpoint] no checkpoint found "
                "(fresh training run)"
            )

            return None

        print(
            f"[checkpoint] restoring "
            f"step {latest_step:,}"
        )

        try:

            # ----------------------------------------------
            # Sharded restore
            # ----------------------------------------------

            restore_args = (
                ocp.args.StandardRestore(
                    item=target_state,
                )
            )

            restored_state = (
                self.manager.restore(
                    latest_step,
                    args=restore_args,
                )
            )

            print(
                "[checkpoint] restore successful:\n"
                f"  step="
                f"{int(restored_state.step):,}\n"
                f"  tokens="
                f"{int(restored_state.tokens_processed):,}"
            )

            return (
                restored_state,
                latest_step,
            )

        except Exception as e:

            print(
                "[checkpoint] RESTORE FAILED:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            print(
                "[checkpoint] "
                "falling back to fresh init"
            )

            return None

    # ─────────────────────────────────────────────
    # Close
    # ─────────────────────────────────────────────

    def close(self):
        """
        Graceful shutdown.
        """

        try:

            self.wait()

        finally:

            self.manager.close()
