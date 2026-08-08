"""
LaughLM/profiling/reports/json.py

Exporter for JSON profile artifacts.
Writes run-level artifacts to profiles/<run_id>/ without per-step I/O overhead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
from LaughLM.profiling.core.session import ProfileSession


def export_json_artifacts(
    session: ProfileSession,
    aggregated: Dict[str, Any],
    diagnostics: Dict[str, Any],
    recommendations: List[Dict[str, str]],
) -> Dict[str, str]:
    """
    Export all JSON artifacts to output directory profiles/<run_id>/.
    Returns dictionary mapping artifact names to absolute path strings.
    """
    out_dir = session.ensure_output_dirs()

    # 1. session.json
    session_data = {
        "run_id": session.run_id,
        "level": session.level,
        "start_wall_time": session.start_wall_time,
        "end_wall_time": session.end_wall_time,
        "total_duration_s": session.total_duration,
        "system_info": session.system_info,
        "config": session.config,
        "total_events_recorded": len(session.events),
        "total_steps_recorded": len(session.step_metrics),
    }
    session_path = out_dir / "session.json"
    with open(session_path, "w") as f:
        json.dump(session_data, f, indent=2)

    # 2. summary.json
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    # 3. diagnostics.json
    diagnostics_path = out_dir / "diagnostics.json"
    with open(diagnostics_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    # 4. recommendations.json
    recs_data = {
        "run_id": session.run_id,
        "primary_bottleneck": diagnostics.get("primary_bottleneck", "unknown"),
        "confidence": diagnostics.get("confidence", "low"),
        "recommendations": recommendations,
    }
    recs_path = out_dir / "recommendations.json"
    with open(recs_path, "w") as f:
        json.dump(recs_data, f, indent=2)

    return {
        "session": str(session_path),
        "summary": str(summary_path),
        "diagnostics": str(diagnostics_path),
        "recommendations": str(recs_path),
    }
