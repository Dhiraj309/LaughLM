"""
LaughLM/profiling/integrations/jax.py

Integration with JAX / XProf profiler APIs.
Provides optional low-level device tracing without making XProf mandatory.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class XProfState(str, Enum):
    """Lifecycle state of the optional JAX/XProf integration."""

    DISABLED = "disabled"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ACTIVE = "active"


class XProfUnavailableError(RuntimeError):
    """Raised when requested XProf tracing cannot be started safely."""


@dataclass(frozen=True)
class XProfCapability:
    """Result of probing the public JAX profiler entry points."""

    state: XProfState
    reason: str = ""


def detect_xprof_capability() -> XProfCapability:
    """Probe XProf support without starting a trace.

    JAX owns the public tracing API.  The actual start call remains part of
    the probe boundary because TPU plugin ABI mismatches are only reported by
    the runtime when tracing is started.
    """

    try:
        import jax
    except Exception as exc:
        return XProfCapability(
            state=XProfState.UNAVAILABLE,
            reason=f"JAX could not be imported: {exc}",
        )

    profiler = getattr(jax, "profiler", None)
    start_trace = getattr(profiler, "start_trace", None)
    stop_trace = getattr(profiler, "stop_trace", None)
    if not callable(start_trace) or not callable(stop_trace):
        return XProfCapability(
            state=XProfState.UNAVAILABLE,
            reason="jax.profiler.start_trace/stop_trace are unavailable",
        )

    return XProfCapability(state=XProfState.AVAILABLE)


_TRACE_ACTIVE = False


def start_jax_trace(log_dir: str) -> XProfState:
    """
    Start JAX profiler trace collection.

    Raises ``XProfUnavailableError`` when the requested trace cannot be
    started.  This is intentionally fail-fast: training must not claim to be
    profiled when the runtime silently rejected the trace.
    """

    global _TRACE_ACTIVE

    capability = detect_xprof_capability()
    if capability.state is not XProfState.AVAILABLE:
        raise XProfUnavailableError(
            f"XProf tracing requested but unavailable: {capability.reason}"
        )

    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        import jax

        jax.profiler.start_trace(log_dir)
    except Exception as exc:
        _TRACE_ACTIVE = False
        raise XProfUnavailableError(
            f"XProf tracing requested but could not start: {exc}"
        ) from exc

    _TRACE_ACTIVE = True
    return XProfState.ACTIVE


def stop_jax_trace() -> bool:
    """
    Stop JAX profiler trace collection.

    The operation is idempotent when no trace is active.
    """

    global _TRACE_ACTIVE
    if not _TRACE_ACTIVE:
        return False

    try:
        import jax

        jax.profiler.stop_trace()
    finally:
        _TRACE_ACTIVE = False
    return True


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
