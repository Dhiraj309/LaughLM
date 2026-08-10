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
    Start JAX profiler trace collection.

    Temporarily disabled because the current TPU runtime has an
    incompatible native profiler plugin:

        PLUGIN_Profiler_Api size: expected 80, got 104

    Returns False without touching the native profiler runtime.
    """
    print(
        "[profiler] JAX/XProf trace disabled for this TPU runtime",
        flush=True,
    )
    return False


def stop_jax_trace() -> bool:
    """
    Stop JAX profiler trace collection.

    No-op while JAX/XProf tracing is disabled.
    """
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
