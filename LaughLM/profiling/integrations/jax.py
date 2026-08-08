"""
LaughLM/profiling/integrations/jax.py

Integration with JAX / XProf profiler APIs.
Provides optional low-level device tracing without making XProf mandatory.
"""

from __future__ import annotations

import contextlib
from typing import Optional


def start_jax_trace(log_dir: str) -> bool:
    """
    Start JAX profiler trace collection (XProf / Perfetto).
    Returns True if trace started successfully, False otherwise.
    """
    try:
        import jax
        jax.profiler.start_trace(log_dir)
        return True
    except Exception as e:
        print(f"[profiler] Info: JAX trace start bypassed ({e})", flush=True)
        return False


def stop_jax_trace() -> bool:
    """
    Stop active JAX profiler trace collection.
    """
    try:
        import jax
        jax.profiler.stop_trace()
        return True
    except Exception as e:
        print(f"[profiler] Info: JAX trace stop bypassed ({e})", flush=True)
        return False


@contextlib.contextmanager
def annotate_section(name: str):
    """
    Annotate JAX execution block with a trace annotation if available.
    """
    try:
        import jax
        if hasattr(jax.profiler, "TraceAnnotation"):
            with jax.profiler.TraceAnnotation(name):
                yield
            return
    except Exception:
        pass
    yield
