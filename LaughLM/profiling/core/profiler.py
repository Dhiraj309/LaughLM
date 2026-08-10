"""
LaughLM/profiling/core/profiler.py

Primary API entrypoint for the LaughLM Performance Profiler.
"""

from __future__ import annotations

import datetime
import time
from typing import Optional, Dict, Any, List

from LaughLM.profiling.core.event import Event
from LaughLM.profiling.core.scope import Scope, NullScope
from LaughLM.profiling.core.session import ProfileSession


class Profiler:
    """
    LaughLM Performance Profiler.

    Provides context-manager section profiling, event aggregation,
    bottleneck diagnosis, and report generation.
    """

    def __init__(
        self,
        enabled: bool = False,
        level: str = "summary",
        output_dir: str = "profiles",
        run_id: Optional[str] = None,
        xprof: bool = False,
        layer_profiling: bool = False,
        warmup_steps: int = 5,
        active_steps: int = 100,
        config: Optional[Any] = None,
    ):
        self.level = (level or "off").lower()
        self.enabled = bool(enabled and self.level != "off")

        if self.level == "off":
            self.enabled = False

        self.output_dir = output_dir
        self.xprof = xprof
        self.layer_profiling = bool(
            layer_profiling or self.level in ("detailed", "developer")
        )
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps

        if not run_id:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}"
        self.run_id = run_id

        self.null_scope = NullScope()
        self.active_stack: List[Event] = []
        self._event_counter = 0

        cfg_dict: Dict[str, Any] = {}
        if config is not None:
            if hasattr(config, "model_dump"):
                cfg_dict = config.model_dump()
            elif isinstance(config, dict):
                cfg_dict = config

        self.session = ProfileSession(
            run_id=self.run_id,
            output_dir=self.output_dir,
            config=cfg_dict,
            level=self.level,
        )

        self._jax_trace_active = False

        if self.enabled and self.xprof and self.level == "developer":
            self._start_xprof_if_requested()

    @classmethod
    def from_config(
        cls,
        config: Any,
        run_id: Optional[str] = None,
        override_enabled: Optional[bool] = None,
        override_level: Optional[str] = None,
        override_output_dir: Optional[str] = None,
    ) -> Profiler:
        """
        Instantiate Profiler from a LaughLMConfig instance or dict.
        """
        prof_cfg = getattr(config, "profiling", None)
        if prof_cfg is None and isinstance(config, dict):
            prof_cfg = config.get("profiling", {})

        if prof_cfg is None:
            return cls(enabled=False, level="off", run_id=run_id, config=config)

        if hasattr(prof_cfg, "enabled"):
            enabled = getattr(prof_cfg, "enabled", False)
            level = getattr(prof_cfg, "level", "summary")
            output_dir = getattr(prof_cfg, "output_dir", "profiles")
            xprof = getattr(prof_cfg, "xprof", False)
            layer_profiling = getattr(prof_cfg, "layer_profiling", False)
            warmup_steps = getattr(prof_cfg, "warmup_steps", 5)
            active_steps = getattr(prof_cfg, "active_steps", 100)
        elif isinstance(prof_cfg, dict):
            enabled = prof_cfg.get("enabled", False)
            level = prof_cfg.get("level", "summary")
            output_dir = prof_cfg.get("output_dir", "profiles")
            xprof = prof_cfg.get("xprof", False)
            layer_profiling = prof_cfg.get("layer_profiling", False)
            warmup_steps = prof_cfg.get("warmup_steps", 5)
            active_steps = prof_cfg.get("active_steps", 100)
        else:
            enabled = False
            level = "off"
            output_dir = "profiles"
            xprof = False
            layer_profiling = False
            warmup_steps = 5
            active_steps = 100

        if override_enabled is not None:
            enabled = override_enabled
        if override_level is not None:
            level = override_level
        if override_output_dir is not None:
            output_dir = override_output_dir

        return cls(
            enabled=enabled,
            level=level,
            output_dir=output_dir,
            run_id=run_id,
            xprof=xprof,
            layer_profiling=layer_profiling,
            warmup_steps=warmup_steps,
            active_steps=active_steps,
            config=config,
        )

    def section(
        self,
        name: str,
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Scope | NullScope:
        """
        Open a timing section. Fast return NullScope when disabled.
        """
        if not self.enabled:
            return self.null_scope
        return Scope(self, name, category=category, metadata=metadata)

    def should_profile_step(self, step_idx: int) -> bool:
        """
        Check if step_idx falls within active profiling window.
        """
        if not self.enabled:
            return False
        if step_idx < self.warmup_steps:
            return False
        if self.active_steps > 0 and step_idx >= (self.warmup_steps + self.active_steps):
            return False
        return True

    def should_profile_layer(self) -> bool:
        """
        Check if layer-level profiling is active.
        """
        return self.enabled and self.layer_profiling

    def _enter_section(
        self,
        name: str,
        category: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Event:
        self._event_counter += 1
        event_id = f"evt_{self._event_counter}"
        parent_name = self.active_stack[-1].name if self.active_stack else None

        event = Event(
            name=name,
            category=category,
            start=time.perf_counter(),
            parent=parent_name,
            metadata=metadata or {},
            event_id=event_id,
        )
        self.active_stack.append(event)
        return event

    def _exit_section(self, event: Optional[Event]) -> None:
        if not event:
            return
        end_time = time.perf_counter()
        event.finish(end_time)
        if self.active_stack and self.active_stack[-1] is event:
            self.active_stack.pop()
        self.session.add_event(event)

    def record_step(
        self,
        step: int,
        duration: float,
        tokens: Optional[int] = None,
        mfu: Optional[float] = None,
        **extra: Any,
    ) -> None:
        """
        Record step-level summary metrics.
        """
        if not self.enabled:
            return
        self.session.record_step_metrics(
            step=step,
            duration=duration,
            tokens=tokens,
            mfu=mfu,
            **extra,
        )

    def _start_xprof_if_requested(self) -> None:
        from LaughLM.profiling.integrations.jax import start_jax_trace
        traces_dir = self.session.ensure_output_dirs() / "traces"
        self._jax_trace_active = start_jax_trace(str(traces_dir))

    def finish(self) -> Dict[str, Any]:
        """
        Finalize profiling session, aggregate timings, diagnose bottlenecks,
        generate Markdown/JSON reports, and print terminal summary.
        """
        if not self.enabled:
            return {}

        self.session.finalize()

        if self._jax_trace_active:
            from LaughLM.profiling.integrations.jax import stop_jax_trace
            stop_jax_trace()
            self._jax_trace_active = False

        from LaughLM.profiling.analysis.aggregation import aggregate_session
        from LaughLM.profiling.analysis.bottlenecks import BottleneckAnalyzer
        from LaughLM.profiling.reports.json import export_json_artifacts
        from LaughLM.profiling.reports.markdown import generate_markdown_report
        from LaughLM.profiling.reports.terminal import render_terminal_report

        aggregated = aggregate_session(self.session)
        analyzer = BottleneckAnalyzer()
        diagnostics = analyzer.analyze(aggregated, self.session)
        recommendations = analyzer.generate_recommendations(diagnostics)

        artifacts = export_json_artifacts(
            session=self.session,
            aggregated=aggregated,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        md_path = generate_markdown_report(
            session=self.session,
            aggregated=aggregated,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        render_terminal_report(
            session=self.session,
            aggregated=aggregated,
            diagnostics=diagnostics,
            recommendations=recommendations,
        )

        return {
            "output_dir": str(self.session.output_dir),
            "artifacts": artifacts,
            "report_md": str(md_path),
            "diagnostics": diagnostics,
            "recommendations": recommendations,
        }
