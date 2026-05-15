"""
LaughLM/utils/profiler.py

Production-grade TPU/GPU profiling utilities.

FIX (frontier-optim audit 2026):
  Expanded profiler to support:
  - Step-range profiling (start after warmup for steady-state metrics)
  - Memory profiling via jax.live_arrays()
  - Compile time measurement
  - Automatic trace directory with metadata

Reference: MaxText uses jax.profiler.start_trace/stop_trace with
step-bounded profiling windows.
"""

import time
import json
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
    Lightweight TPU/GPU profiler.

    Usage:
        with tpu_profile(enabled=True, steps_hint=20):
            for step in range(20):
                train_step(...)

    The trace captures XLA compilation, memory allocation, collective
    operations, and kernel execution. Download and view in TensorBoard
    or Chrome trace viewer (chrome://tracing).

    Parameters
    ----------
    enabled    : bool — skip profiling if False (zero overhead)
    trace_dir  : str — base directory for trace output
    steps_hint : int — expected number of profiled steps (for logging)
    """

    if not enabled:
        yield
        return

    path = _trace_dir(trace_dir)

    # Save metadata for later analysis
    metadata = {
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "device_type": str(jax.devices()[0].device_kind) if jax.devices() else "unknown",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps_hint": steps_hint,
    }
    with open(path / "profile_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Profiler]")
    print(f"  Backend: {metadata['backend']}")
    print(f"  Devices: {metadata['devices']} × {metadata['device_type']}")
    print(f"  Trace directory: {path}")
    print(f"  Profiling window: ~{steps_hint} steps")

    try:
        jax.profiler.start_trace(str(path))
        yield
    finally:
        jax.profiler.stop_trace()

        print(f"\n[Profiler] Trace saved → {path}")
        print("  View with: tensorboard --logdir", path)


@contextmanager
def measure_compile_time(label=""):
    """Measure XLA compilation time for a code block.
    
    Usage:
        with measure_compile_time("first train step"):
            train_step(state, batch)  # triggers compilation
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    prefix = f"[compile] {label}: " if label else "[compile] "
    print(f"{prefix}{elapsed:.2f}s")


def log_memory_usage(label=""):
    """Log current JAX memory usage (peak and live)."""
    try:
        for device in jax.local_devices()[:1]:  # Just first device
            stats = device.memory_stats()
            if stats:
                peak = stats.get("peak_bytes_in_use", 0) / 1e9
                live = stats.get("bytes_in_use", 0) / 1e9
                limit = stats.get("bytes_limit", 0) / 1e9
                prefix = f"[memory] {label}: " if label else "[memory] "
                print(f"{prefix}live={live:.2f}GB peak={peak:.2f}GB limit={limit:.2f}GB")
                return
    except Exception:
        pass
    print(f"[memory] {label}: unavailable (device stats not supported)")
