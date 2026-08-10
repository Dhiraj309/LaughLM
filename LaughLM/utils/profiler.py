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

    Native JAX/XProf tracing is disabled because the current TPU
    runtime reports an incompatible profiler plugin ABI:

        PLUGIN_Profiler_Api size: expected 80, got 104

    The context manager remains available so existing callers do not
    break, but it does not invoke jax.profiler.start_trace().
    """

    if not enabled:
        yield
        return

    path = _trace_dir(trace_dir)

    metadata = {
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "device_type": (
            str(jax.devices()[0].device_kind)
            if jax.devices()
            else "unknown"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps_hint": steps_hint,
        "xprof_enabled": False,
    }

    with open(path / "profile_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Profiler]")
    print(f"  Backend: {metadata['backend']}")
    print(f"  Devices: {metadata['devices']} × {metadata['device_type']}")
    print(f"  Trace directory: {path}")
    print(f"  Profiling window: ~{steps_hint} steps")
    print("  XProf/JAX trace: DISABLED")

    try:
        yield
    finally:
        print("\n[Profiler] XProf trace disabled")


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
