"""
LaughLM/profiling/core/session.py

Profile session manager that aggregates events and serializes run artifacts.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from LaughLM.profiling.core.event import Event


class ProfileSession:
    """
    Session container for a single profiling run.
    Collects events, tracks step throughput/MFU metrics, and manages run artifacts.
    """

    def __init__(
        self,
        run_id: str,
        output_dir: str = "profiles",
        config: Optional[Dict[str, Any]] = None,
        level: str = "summary",
    ):
        self.run_id = run_id
        self.root_output_dir = Path(output_dir)
        self.output_dir = self.root_output_dir / run_id
        self.config = config or {}
        self.level = level

        self.start_wall_time = time.time()
        self.start_perf_time = time.perf_counter()
        self.end_wall_time: Optional[float] = None
        self.end_perf_time: Optional[float] = None

        self.events: List[Event] = []
        self.step_metrics: List[Dict[str, Any]] = []

        self.system_info = self._gather_system_info()

    def _gather_system_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "python_version": sys.version,
            "platform": sys.platform,
        }
        try:
            import jax
            info["jax_version"] = getattr(jax, "__version__", "unknown")
            info["jax_backend"] = jax.default_backend()
            info["device_count"] = jax.device_count()
            info["local_device_count"] = jax.local_device_count()
            devices = [str(d) for d in jax.local_devices()]
            info["local_devices"] = devices
        except Exception as e:
            info["jax_error"] = str(e)
        return info

    def add_event(self, event: Event) -> None:
        """
        Record a timing event.
        """
        self.events.append(event)

    def record_step_metrics(
        self,
        step: int,
        duration: float,
        tokens: Optional[int] = None,
        mfu: Optional[float] = None,
        tokens_per_sec: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """
        Record top-level metrics for a single training step.
        """
        if tokens_per_sec is None and tokens is not None and duration > 0:
            tokens_per_sec = tokens / duration

        entry = {
            "step": step,
            "duration": duration,
            "duration_ms": duration * 1000.0,
            "tokens": tokens,
            "tokens_per_sec": tokens_per_sec,
            "mfu": mfu,
            **extra,
        }
        self.step_metrics.append(entry)

    def finalize(self) -> None:
        """
        Mark session as complete.
        """
        if self.end_wall_time is None:
            self.end_wall_time = time.time()
            self.end_perf_time = time.perf_counter()

    @property
    def total_duration(self) -> float:
        """
        Total session duration in seconds.
        """
        if self.end_perf_time is not None:
            return max(0.0, self.end_perf_time - self.start_perf_time)
        return max(0.0, time.perf_counter() - self.start_perf_time)

    def ensure_output_dirs(self) -> Path:
        """
        Ensure output directories exist.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = self.output_dir / "traces"
        traces_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
