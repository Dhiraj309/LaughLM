"""
LaughLM/profiling/core/scope.py

Context manager scopes for profiling sections.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from LaughLM.profiling.core.profiler import Profiler
    from LaughLM.profiling.core.event import Event


class Scope:
    """
    Context manager for a profiling section.
    """

    def __init__(
        self,
        profiler: Profiler,
        name: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.profiler = profiler
        self.name = name
        self.category = category
        self.metadata = metadata or {}
        self.event: Optional[Event] = None

    def __enter__(self) -> Scope:
        self.event = self.profiler._enter_section(
            self.name,
            self.category,
            self.metadata,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.profiler._exit_section(self.event)
        return False


class NullScope:
    """
    Zero-overhead no-op context manager when profiler is disabled.
    """

    def __enter__(self) -> NullScope:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False
