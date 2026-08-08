"""
LaughLM/profiling/reports/terminal.py

Terminal report renderer for terminal / console output.
"""

from __future__ import annotations

from typing import Dict, Any, List
from LaughLM.profiling.core.session import ProfileSession


def render_terminal_report(
    session: ProfileSession,
    aggregated: Dict[str, Any],
    diagnostics: Dict[str, Any],
    recommendations: List[Dict[str, str]],
) -> None:
    """
    Render a clean summary table to stdout.
    """
    run_id = session.run_id
    level = session.level
    mean_step_ms = aggregated.get("mean_step_ms", 0.0)
    tok_sec = aggregated.get("mean_tokens_per_sec", 0.0)
    mfu = aggregated.get("mean_mfu", 0.0)
    total_steps = aggregated.get("total_steps", 0)

    primary = diagnostics.get("primary_bottleneck", "unknown")
    confidence = diagnostics.get("confidence", "low")

    print("\n" + "=" * 70, flush=True)
    print(f" LaughLM Performance Profiler Summary [{run_id}] (level: {level})", flush=True)
    print("=" * 70, flush=True)

    print(f" Steps Profiled  : {total_steps:,}", flush=True)
    print(f" Mean Step Time  : {mean_step_ms:.2f} ms", flush=True)
    if tok_sec > 0:
        print(f" Throughput      : {tok_sec:,.1f} tokens/sec", flush=True)
    if mfu > 0:
        print(f" MFU             : {mfu * 100:.2f}%", flush=True)

    print(f"\n Primary Bottleneck: {primary.upper()} (Confidence: {confidence})", flush=True)
    if diagnostics.get("evidence"):
        print(f" Evidence          : {diagnostics['evidence']}", flush=True)

    print("-" * 70, flush=True)
    print(" Step Time Breakdown by Category:", flush=True)
    print(f" {'Category':<20} | {'Mean (ms)':<10} | {'% of Step':<10} | {'Count':<8}", flush=True)
    print(" " + "-" * 66, flush=True)

    by_cat = aggregated.get("by_category", {})
    sorted_cats = sorted(
        by_cat.items(),
        key=lambda x: x[1].get("total_ms", 0.0),
        reverse=True,
    )

    for cat_name, stats in sorted_cats:
        print(
            f" {cat_name:<20} | {stats['mean_ms']:<10.2f} | {stats['pct_of_step']:<9.1f}% | {stats['count']:<8}",
            flush=True,
        )

    if recommendations:
        print("-" * 70, flush=True)
        print(" Actionable Recommendations:", flush=True)
        for i, rec in enumerate(recommendations, 1):
            print(f" [{i}] {rec['bottleneck']}:", flush=True)
            print(f"     Recommendation : {rec['recommendation']}", flush=True)

    out_dir = session.output_dir
    print("=" * 70, flush=True)
    print(f" Run Artifacts Saved To: {out_dir}", flush=True)
    print("=" * 70 + "\n", flush=True)
