"""
LaughLM/profiling/reports/markdown.py

Markdown report generator for performance profiles.
Generates report.md within output_dir/<run_id>/.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Dict, Any, List
from LaughLM.profiling.core.session import ProfileSession


def generate_markdown_report(
    session: ProfileSession,
    aggregated: Dict[str, Any],
    diagnostics: Dict[str, Any],
    recommendations: List[Dict[str, str]],
) -> Path:
    """
    Generate Markdown performance profile report.md.
    """
    out_dir = session.ensure_output_dirs()
    report_path = out_dir / "report.md"

    dt_str = datetime.datetime.fromtimestamp(session.start_wall_time).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    primary = diagnostics.get("primary_bottleneck", "unknown")
    confidence = diagnostics.get("confidence", "low")
    evidence = diagnostics.get("evidence", "")

    mean_step_ms = aggregated.get("mean_step_ms", 0.0)
    tok_sec = aggregated.get("mean_tokens_per_sec", 0.0)
    mfu = aggregated.get("mean_mfu", 0.0)
    total_steps = aggregated.get("total_steps", 0)

    lines: List[str] = []

    lines.append(f"# LaughLM Performance Profile Report")
    lines.append("")
    lines.append(f"- **Run ID:** `{session.run_id}`")
    lines.append(f"- **Profile Level:** `{session.level}`")
    lines.append(f"- **Timestamp:** {dt_str}")
    lines.append(f"- **Steps Profiled:** {total_steps}")
    lines.append(f"- **Session Duration:** {session.total_duration:.2f} s")
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- **Primary Bottleneck:** `{primary}` (Confidence: `{confidence}`)")
    lines.append(f"- **Evidence:** {evidence}")
    lines.append(f"- **Mean Step Time:** `{mean_step_ms:.2f} ms`")
    if tok_sec > 0:
        lines.append(f"- **Throughput:** `{tok_sec:,.1f} tokens/sec`")
    if mfu > 0:
        lines.append(f"- **Model Flops Utilization (MFU):** `{mfu * 100:.2f}%`")
    lines.append("")

    lines.append("## Step Breakdown by Category")
    lines.append("")
    lines.append("| Category | Mean Time (ms) | % of Step | Count | Min (ms) | Max (ms) | p95 (ms) |")
    lines.append("|---|---|---|---|---|---|---|")

    by_category = aggregated.get("by_category", {})
    sorted_cats = sorted(
        by_category.items(),
        key=lambda x: x[1].get("total_ms", 0.0),
        reverse=True,
    )

    for cat_name, stats in sorted_cats:
        lines.append(
            f"| `{cat_name}` | {stats['mean_ms']:.2f} | {stats['pct_of_step']:.1f}% | {stats['count']} | {stats['min_ms']:.2f} | {stats['max_ms']:.2f} | {stats['p95_ms']:.2f} |"
        )
    lines.append("")

    lines.append("## Top Invocations by Event Name")
    lines.append("")
    lines.append("| Event Name | Mean Time (ms) | Total Time (ms) | Count | p90 (ms) |")
    lines.append("|---|---|---|---|---|")

    by_name = aggregated.get("by_name", {})
    sorted_names = sorted(
        by_name.items(),
        key=lambda x: x[1].get("total_ms", 0.0),
        reverse=True,
    )[:15]

    for event_name, stats in sorted_names:
        lines.append(
            f"| `{event_name}` | {stats['mean_ms']:.2f} | {stats['total_ms']:.2f} | {stats['count']} | {stats['p90_ms']:.2f} |"
        )
    lines.append("")

    lines.append("## Bottleneck Analysis")
    lines.append("")
    findings = diagnostics.get("findings", [])
    if not findings:
        lines.append("No critical bottleneck thresholds were breached.")
    else:
        for finding in findings:
            lines.append(f"### Bottleneck: `{finding['bottleneck']}`")
            lines.append(f"- **Category:** {finding.get('category', 'N/A')}")
            lines.append(f"- **Confidence:** `{finding['confidence']}`")
            lines.append(f"- **Evidence:** {finding['evidence']}")
            lines.append("")

    lines.append("## Actionable Optimization Recommendations")
    lines.append("")
    if not recommendations:
        lines.append("No immediate optimizations recommended based on current measurements.")
    else:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"### {i}. `{rec['bottleneck']}`")
            lines.append(f"**Recommendation:** {rec['recommendation']}")
            lines.append(f"**Rationale:** {rec['rationale']}")
            lines.append("")

    lines.append("## Environment & System Info")
    lines.append("")
    sys_info = session.system_info
    for k, v in sys_info.items():
        lines.append(f"- **{k}:** `{v}`")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path
