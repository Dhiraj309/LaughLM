"""
LaughLM/profiling/analysis/aggregation.py

Aggregation utilities for computing statistical breakdowns of profiling events.
"""

from __future__ import annotations

import math
from typing import Dict, Any, List
from LaughLM.profiling.core.session import ProfileSession
from LaughLM.profiling.core.event import Event


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_v[int(k)]
    d0 = sorted_v[int(f)] * (c - k)
    d1 = sorted_v[int(c)] * (k - f)
    return d0 + d1


def _compute_stats(durations_ms: List[float], total_benchmark_ms: float) -> Dict[str, Any]:
    if not durations_ms:
        return {
            "count": 0,
            "total_ms": 0.0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "pct_of_step": 0.0,
        }

    count = len(durations_ms)
    total_ms = sum(durations_ms)
    mean_ms = total_ms / count
    var_ms = sum((x - mean_ms) ** 2 for x in durations_ms) / count if count > 1 else 0.0
    std_ms = math.sqrt(var_ms)

    pct = (total_ms / total_benchmark_ms * 100.0) if total_benchmark_ms > 0 else 0.0

    return {
        "count": count,
        "total_ms": total_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(durations_ms),
        "max_ms": max(durations_ms),
        "p50_ms": _percentile(durations_ms, 50),
        "p90_ms": _percentile(durations_ms, 90),
        "p95_ms": _percentile(durations_ms, 95),
        "pct_of_step": pct,
    }


def aggregate_session(session: ProfileSession) -> Dict[str, Any]:
    """
    Aggregate all recorded session events and step statistics.
    """
    events = session.events
    step_metrics = session.step_metrics

    # Group event durations by name and category
    by_name: Dict[str, List[float]] = {}
    by_category: Dict[str, List[float]] = {}

    for event in events:
        dur_ms = event.duration * 1000.0
        by_name.setdefault(event.name, []).append(dur_ms)
        by_category.setdefault(event.category, []).append(dur_ms)

    # Compute step benchmark total time
    step_durations_ms = [
        s["duration_ms"] for s in step_metrics if "duration_ms" in s
    ]
    if not step_durations_ms and "step" in by_name:
        step_durations_ms = by_name["step"]

    total_step_benchmark_ms = sum(step_durations_ms) if step_durations_ms else 0.0
    mean_step_ms = (
        sum(step_durations_ms) / len(step_durations_ms) if step_durations_ms else 0.0
    )

    # Aggregations by name
    name_stats = {
        name: _compute_stats(durs, total_step_benchmark_ms)
        for name, durs in by_name.items()
    }

    # Aggregations by category
    category_stats = {
        cat: _compute_stats(durs, total_step_benchmark_ms)
        for cat, durs in by_category.items()
    }

    # Extract throughput and MFU metrics
    tok_sec_list = [
        s["tokens_per_sec"]
        for s in step_metrics
        if s.get("tokens_per_sec") is not None
    ]
    mfu_list = [s["mfu"] for s in step_metrics if s.get("mfu") is not None]

    mean_tok_sec = sum(tok_sec_list) / len(tok_sec_list) if tok_sec_list else 0.0
    mean_mfu = sum(mfu_list) / len(mfu_list) if mfu_list else 0.0

    return {
        "run_id": session.run_id,
        "level": session.level,
        "total_session_duration_s": session.total_duration,
        "total_steps": len(step_durations_ms),
        "mean_step_ms": mean_step_ms,
        "mean_tokens_per_sec": mean_tok_sec,
        "mean_mfu": mean_mfu,
        "by_name": name_stats,
        "by_category": category_stats,
        "step_metrics": step_metrics,
    }
