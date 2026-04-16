import time
from pathlib import Path
import orbax.checkpoint as ocp


class CheckpointManager:
    """
    Orbax checkpoint manager for LaughLM.

    - Async checkpointing (fast during training)
    - Safe finalization (no shutdown crashes)
    - Resume support
    """

    def __init__(self, directory: str, max_to_keep: int = 3, async_enabled: bool = True):

        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------
        # Async config (disable if debugging)
        # ------------------------------------------------------------
        async_options = None
        if not async_enabled:
            async_options = ocp.AsyncOptions(enable_async=False)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
            async_options=async_options,
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state",),
            options=options,
        )

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------

    def save(self, step: int, state, wait: bool = False):

        print(f"[checkpoint] saving step {step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state)
        )

        # self.manager.save(step, args=args)
        
        if not self.manager.is_saving_in_progress():
            self.manager.save(step, args=args)
            
        else:
            print(f"[checkpoint] skipped step {step} (previous save still running)")

        # ------------------------------------------------------------
        # Final save → block + flush async tasks
        # ------------------------------------------------------------
        if wait:
            self.wait_until_finished()

            # 🔥 Important for Orbax async metadata (Kaggle fix)
            time.sleep(2)

            print(f"[checkpoint] saved step {step}")

    # ------------------------------------------------------------
    # Wait for async saves to complete
    # ------------------------------------------------------------

    def wait_until_finished(self):
        """Block until all pending checkpoint writes are complete."""
        self.manager.wait_until_finished()

    # ------------------------------------------------------------
    # Clean shutdown (fixes your AttributeError + async crash)
    # ------------------------------------------------------------

    def close(self):
        """Ensure all checkpoint operations are fully finished."""
        print("[checkpoint] closing manager...")
        self.wait_until_finished()
        time.sleep(2)  # allow final async metadata writes
        print("[checkpoint] all writes finished")

    # ------------------------------------------------------------
    # Restore latest checkpoint
    # ------------------------------------------------------------

    def restore_latest(self, target_state=None):

        latest_step = self.manager.latest_step()

        if latest_step is None:
            return None

        print(f"[checkpoint] restoring step {latest_step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardRestore(item=target_state)
        )

        restored = self.manager.restore(
            latest_step,
            args=args,
        )

        return restored["state"], latest_step