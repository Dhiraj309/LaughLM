import orbax.checkpoint as ocp
from pathlib import Path


class CheckpointManager:
    """
    Modern Orbax checkpoint manager for LaughLM.

    Saves complete training state for seamless resume.
    """

    def __init__(self, directory: str, max_to_keep: int = 3):

        self.directory = Path(directory).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=True,
        )

        self.manager = ocp.CheckpointManager(
            self.directory,
            item_names=("state",),
            options=options,
        )

    # ------------------------------------------------------------
    # Save checkpoint
    # ------------------------------------------------------------

    def save(self, step: int, state: dict):

        print(f"[checkpoint] saving step {step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardSave(state)
        )

        self.manager.save(step, args=args)

        print(f"[checkpoint] saved step {step}")

    # ------------------------------------------------------------
    # Restore latest checkpoint
    # ------------------------------------------------------------

    def restore_latest(self):

        latest_step = self.manager.latest_step()

        if latest_step is None:
            return None

        print(f"[checkpoint] restoring step {latest_step}")

        args = ocp.args.Composite(
            state=ocp.args.StandardRestore()
        )

        restored = self.manager.restore(latest_step, args=args)

        return restored["state"], latest_step
