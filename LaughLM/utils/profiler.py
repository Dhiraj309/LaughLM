import time
import jax
from pathlib import Path
from contextlib import contextmanager


def _trace_dir(base_dir="tpu_traces"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = Path(base_dir) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def tpu_profile(enabled=True, trace_dir="tpu_traces", steps_hint=20):
    """
    Lightweight TPU profiler suitable for Kaggle.
    """

    if not enabled:
        yield
        return

    path = _trace_dir(trace_dir)

    print("\n[TPU Profiler]")
    print(f"Trace directory: {path}")
    print(f"Recommended profiling window: ~{steps_hint} steps")

    try:
        jax.profiler.start_trace(str(path))
        yield
    finally:
        jax.profiler.stop_trace()

        print("\nTrace saved.")
        print("Download directory and open with Chrome trace viewer.")
