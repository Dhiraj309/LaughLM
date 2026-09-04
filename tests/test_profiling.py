"""
tests/test_profiling.py

Unit tests for the LaughLM Performance Profiler.
No TPU hardware required.
"""

import os
import tempfile
import time
import pytest

from LaughLM.profiling.core.event import Event
from LaughLM.profiling.core.scope import Scope, NullScope
from LaughLM.profiling.core.session import ProfileSession
from LaughLM.profiling.core.profiler import Profiler
from LaughLM.profiling.analysis.aggregation import aggregate_session
from LaughLM.profiling.analysis.bottlenecks import BottleneckAnalyzer
from LaughLM.profiling.reports.json import export_json_artifacts
from LaughLM.profiling.reports.markdown import generate_markdown_report
from LaughLM.profiling.reports.terminal import render_terminal_report
from LaughLM.profiling.integrations import (
    XProfCapability,
    XProfState,
    XProfUnavailableError,
)
import LaughLM.profiling.integrations.jax as jax_integration


def test_event_creation_and_nesting():
    profiler = Profiler(enabled=True, level="summary", run_id="test_nesting")

    with profiler.section("outer_step", category="step") as outer:
        time.sleep(0.01)
        with profiler.section("data_wait", category="data") as inner:
            time.sleep(0.005)

    events = profiler.session.events
    assert len(events) == 2

    # Check order (inner finishes first in scope __exit__)
    inner_evt = [e for e in events if e.name == "data_wait"][0]
    outer_evt = [e for e in events if e.name == "outer_step"][0]

    assert inner_evt.parent == "outer_step"
    assert inner_evt.duration > 0.0
    assert outer_evt.duration >= inner_evt.duration


def test_disabled_profiler():
    profiler = Profiler(enabled=False, level="off", run_id="test_disabled")
    assert not profiler.enabled

    scope = profiler.section("some_section")
    assert isinstance(scope, NullScope)

    with scope:
        pass

    assert len(profiler.session.events) == 0
    artifacts = profiler.finish()
    assert artifacts == {}


def test_disabled_xprof_is_a_no_op(monkeypatch):
    def fail_if_started(_log_dir):
        raise AssertionError("disabled profiling must not start XProf")

    monkeypatch.setattr(jax_integration, "start_jax_trace", fail_if_started)
    profiler = Profiler(enabled=False, level="off", xprof=True, run_id="no_xprof")

    assert profiler.xprof_state is XProfState.DISABLED
    assert profiler.finish() == {}


def test_requested_unavailable_xprof_fails_before_training(monkeypatch):
    def unavailable(_log_dir):
        raise XProfUnavailableError("test ABI mismatch")

    monkeypatch.setattr(jax_integration, "start_jax_trace", unavailable)

    with pytest.raises(XProfUnavailableError, match="test ABI mismatch"):
        Profiler(enabled=True, level="summary", xprof=True, run_id="bad_xprof")


def test_xprof_state_transitions_to_active_and_back(monkeypatch, tmp_path):
    calls = []
    fake_jax = type(
        "FakeJax",
        (),
        {
            "profiler": type(
                "FakeProfiler",
                (),
                {
                    "start_trace": staticmethod(
                        lambda path: calls.append(("start", path))
                    ),
                    "stop_trace": staticmethod(lambda: calls.append(("stop",))),
                },
            )()
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "jax", fake_jax)
    monkeypatch.setattr(
        jax_integration,
        "detect_xprof_capability",
        lambda: XProfCapability(XProfState.AVAILABLE),
    )

    profiler = Profiler(
        enabled=True,
        level="summary",
        xprof=True,
        output_dir=str(tmp_path),
        run_id="active_xprof",
    )
    assert profiler.xprof_state is XProfState.ACTIVE
    assert profiler._jax_trace_active
    assert calls[0][0] == "start"

    profiler.finish()
    assert profiler.xprof_state is XProfState.AVAILABLE
    assert calls[-1][0] == "stop"


def test_aggregation():
    session = ProfileSession(run_id="test_agg")

    # Add artificial events
    for step in range(5):
        evt_step = Event(name="step", category="step", start=0.0, end=0.1, duration=0.1)
        evt_data = Event(name="data_wait", category="data", start=0.0, end=0.03, duration=0.03, parent="step")
        session.add_event(evt_step)
        session.add_event(evt_data)
        session.record_step_metrics(step=step, duration=0.1, tokens=1000, mfu=0.25)

    session.finalize()
    agg = aggregate_session(session)

    assert agg["total_steps"] == 5
    assert pytest.approx(agg["mean_step_ms"], rel=1e-2) == 100.0
    assert "data" in agg["by_category"]
    assert pytest.approx(agg["by_category"]["data"]["mean_ms"], rel=1e-2) == 30.0


def test_bottleneck_detection_pipeline_bound():
    session = ProfileSession(run_id="test_pipeline")

    for step in range(5):
        session.add_event(Event(name="step", category="step", duration=0.1))
        session.add_event(Event(name="data_wait", category="data", duration=0.04)) # 40% of step
        session.record_step_metrics(step=step, duration=0.1)

    session.finalize()
    agg = aggregate_session(session)

    analyzer = BottleneckAnalyzer()
    diag = analyzer.analyze(agg, session)
    recs = analyzer.generate_recommendations(diag)

    assert diag["primary_bottleneck"] == "pipeline-bound"
    assert diag["confidence"] in ("high", "medium")
    assert len(recs) >= 1
    assert recs[0]["bottleneck"] == "pipeline-bound"


def test_bottleneck_detection_compile_bound():
    session = ProfileSession(run_id="test_compile")

    # Step 0 takes 1.0s, subsequent steps take 0.1s
    session.record_step_metrics(step=0, duration=1.0)
    for step in range(1, 5):
        session.record_step_metrics(step=step, duration=0.1)

    session.finalize()
    agg = aggregate_session(session)

    analyzer = BottleneckAnalyzer()
    diag = analyzer.analyze(agg, session)

    findings = diag["findings"]
    compile_findings = [f for f in findings if f["bottleneck"] == "compile-bound"]
    assert len(compile_findings) == 1
    assert compile_findings[0]["confidence"] == "high"


def test_report_generation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        session = ProfileSession(run_id="test_report_run", output_dir=tmp_dir)

        evt_step = Event(name="step", category="step", duration=0.05)
        evt_compute = Event(name="device_step", category="compute", duration=0.04, parent="step")
        session.add_event(evt_step)
        session.add_event(evt_compute)
        session.record_step_metrics(step=0, duration=0.05, tokens=512, mfu=0.30)
        session.finalize()

        agg = aggregate_session(session)
        analyzer = BottleneckAnalyzer()
        diag = analyzer.analyze(agg, session)
        recs = analyzer.generate_recommendations(diag)

        json_paths = export_json_artifacts(session, agg, diag, recs)
        md_path = generate_markdown_report(session, agg, diag, recs)
        render_terminal_report(session, agg, diag, recs)

        for name, path in json_paths.items():
            assert os.path.exists(path), f"JSON artifact missing: {name} at {path}"

        assert os.path.exists(md_path), "Markdown report missing"
        with open(md_path, "r") as f:
            content = f.read()
            assert "# LaughLM Performance Profile Report" in content
            assert "test_report_run" in content


def test_missing_and_empty_metrics_handling():
    session = ProfileSession(run_id="empty_run")
    session.finalize()

    agg = aggregate_session(session)
    assert agg["total_steps"] == 0
    assert agg["mean_step_ms"] == 0.0

    analyzer = BottleneckAnalyzer()
    diag = analyzer.analyze(agg, session)
    assert diag["primary_bottleneck"] == "unknown"

    recs = analyzer.generate_recommendations(diag)
    assert len(recs) == 1
    assert recs[0]["bottleneck"] == "unknown"
