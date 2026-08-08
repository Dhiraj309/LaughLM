"""
LaughLM/profiling/core/event.py

Representation of a timed execution event within the profiler hierarchy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Event:
    """
    Individual timing event recorded by the profiler.
    """

    name: str
    category: str = "general"
    start: float = field(default_factory=time.perf_counter)
    end: Optional[float] = None
    duration: float = 0.0
    parent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""

    def finish(self, end_time: Optional[float] = None) -> float:
        """
        Record the completion timestamp and compute event duration.
        """
        if end_time is None:
            end_time = time.perf_counter()
        self.end = end_time
        self.duration = max(0.0, self.end - self.start)
        return self.duration

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert event to a dictionary format for serialization.
        """
        return {
            "name": self.name,
            "category": self.category,
            "start": self.start,
            "end": self.end,
            "duration": self.duration,
            "duration_ms": self.duration * 1000.0,
            "parent": self.parent,
            "metadata": self.metadata,
            "event_id": self.event_id,
        }
