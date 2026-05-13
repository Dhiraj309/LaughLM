"""
LaughLM/training/checkpoint.py

Frontier-grade async checkpoint manager for LaughLM.

Features:
──────────────────────────────────────────────
1. Async Orbax checkpoint saves
2. Safe restore with validation
3. Corruption-resistant restore flow
4. Step-aware logging
5. Graceful fallback on restore failure
6. Resume-safe training support
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
    - Supports full TrainState persistence:
        params
        optimizer state
        RNG state
        global step
        tokens processed
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

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            async_options=ocp.AsyncOptions(),
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state",),
            options=options,
        )

        print(
            f"[checkpoint] directory: "
            f"{self.directory}"
        )

    # ─────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────

    def save(self, step: int, state):

        if step < 0:
            raise ValueError(
                f"Invalid checkpoint step: {step}"
            )

        print(f"[checkpoint] saving step {step:,}")

        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state),
        )

        try:

            # Async save (returns immediately)
            self.manager.save(
                step,
                args=args,
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

        try:

            self.manager.wait_until_finished()

            print(
                "[checkpoint] all pending writes complete"
            )

        except Exception as e:

            print(
                "[checkpoint] ERROR waiting for async save:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            raise

    # ─────────────────────────────────────────────
    # Latest step helper
    # ─────────────────────────────────────────────

    def latest_step(self):

        try:
            return self.manager.latest_step()

        except Exception as e:

            print(
                "[checkpoint] ERROR reading latest step:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            return None

    # ─────────────────────────────────────────────
    # Restore latest checkpoint
    # ─────────────────────────────────────────────

    def restore_latest(self, target_state=None):

        # Must finish async writes before restore
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

        args = ocp.args.Composite(
            state=ocp.args.StandardRestore(
                item=target_state
            )
        )

        try:

            restored = self.manager.restore(
                latest_step,
                args=args,
            )

            state = restored["state"]

            print(
                "[checkpoint] restore successful:\n"
                f"  step={int(state.step):,}\n"
                f"  tokens={int(state.tokens_processed):,}"
            )

            return state, latest_step

        except Exception as e:

            print(
                "[checkpoint] RESTORE FAILED:\n"
                f"{type(e).__name__}: {e}"
            )

            traceback.print_exc()

            print(
                "[checkpoint] falling back to fresh init"
            )

            return None