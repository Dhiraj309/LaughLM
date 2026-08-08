"""
LaughLM/profiling/analysis/comparison.py

Utilities for comparing performance profile sessions.
"""

from __future__ import annotations

from typing import Dict, Any
from LaughLM.profiling.core.session import ProfileSession
from LaughLM.profiling.analysis.aggregation import aggregate_session


def compare_sessions(
    session_baseline: ProfileSession,
    session_current: ProfileSession,
) -> Dict[str, Any]:
    """
    Compare two profile sessions and compute performance delta metrics.
    """
    agg_base = aggregate_session(session_baseline)
    agg_curr = aggregate_session(session_current)

    base_step_ms = agg_base.get("mean_step_ms", 0.0)
    curr_step_ms = agg_curr.get("mean_step_ms", 0.0)

    step_delta_ms = curr_step_ms - base_step_ms
    step_pct_change = (
        ((curr_step_ms - base_step_ms) / base_step_ms * 100.0)
        if base_step_ms > 0
        else 0.0
    )

    base_tok = agg_base.get("mean_tokens_per_sec", 0.0)
    curr_tok = agg_curr.get("mean_tokens_per_sec", 0.0)
    tok_pct_change = (
        ((curr_tok - base_tok) / base_tok * 100.0)
        if base_tok > 0
        else 0.0
    )

    # Compare categories
    cats_base = agg_base.get("by_category", {})
    cats_curr = agg_curr.get("by_category", {})

    category_deltas = {}
    all_cats = set(cats_base.keys()) | set(cats_curr.keys())

    for cat in all_cats:
        b_ms = cats_base.get(cat, {}).get("mean_ms", 0.0)
        c_ms = cats_curr.get(cat, {}).get("mean_ms", 0.0)
        delta = c_ms - b_ms
        pct = ((c_ms - b_ms) / b_ms * 100.0) if b_ms > 0 else 0.0
        category_deltas[cat] = {
            "baseline_mean_ms": b_ms,
            "current_mean_ms": c_ms,
            "delta_ms": delta,
            "pct_change": pct,
        }

    return {
        "baseline_run_id": session_baseline.run_id,
        "current_run_id": session_current.run_id,
        "baseline_mean_step_ms": base_step_ms,
        "current_mean_step_ms": curr_step_ms,
        "step_delta_ms": step_delta_ms,
        "step_pct_change": step_pct_change,
        "baseline_tokens_per_sec": base_tok,
        "current_tokens_per_sec": curr_tok,
        "tokens_per_sec_pct_change": tok_pct_change,
        "category_deltas": category_deltas,
    }
