#!/usr/bin/env python3
"""Generate a reproducible static report from a LaughLM metrics artifact.

This script intentionally imports no JAX, model, or accelerator runtime. It is
safe to run after a TPU job from the saved checkpoint directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from LaughLM.analysis.metrics import resolve_metrics_path, summarize_metrics


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}{suffix}"


def _pct(value: Any, total: Any) -> str:
    if value is None or total in (None, 0):
        return "n/a"
    return f"{100.0 * float(value) / float(total):.1f}%"


def _manifest_summary(manifest: dict[str, Any] | None) -> list[str]:
    if not manifest:
        return ["- Run manifest: `not found`"]

    lines = [
        f"- Git revision: `{manifest.get('git_revision') or 'unknown'}`",
        f"- Config path: `{manifest.get('config_path') or 'unknown'}`",
    ]
    jax_info = manifest.get("jax", {})
    if isinstance(jax_info, dict):
        lines.extend(
            [
                f"- Process topology: `{jax_info.get('process_index', 'n/a')}/{jax_info.get('process_count', 'n/a')}`",
                f"- Local devices: `{jax_info.get('local_device_count', 'n/a')}`",
                f"- JAX x64 enabled: `{jax_info.get('x64_enabled', 'unknown')}`",
            ]
        )
    cache_info = manifest.get("compilation_cache", {})
    if isinstance(cache_info, dict):
        cache_state = (
            "warm candidate"
            if cache_info.get("file_count_before_run", 0) > 0
            else "cold candidate"
        )
        lines.extend(
            [
                f"- Compilation cache: `{cache_state}`",
                f"- Cache directory: `{cache_info.get('directory') or 'disabled'}`",
                f"- Cache files before run: `{cache_info.get('file_count_before_run', 'n/a')}`",
            ]
        )
    return lines


def build_report(
    *,
    metrics_path: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any] | None,
    checkpoint_timings: list[dict[str, Any]],
    skip_steps: int,
    last_n: int | None,
) -> str:
    total_step = summary.get("total_step_time_mean")
    checkpoint_total = sum(
        float(record.get("total_overhead_time", 0.0))
        for record in checkpoint_timings
    )
    checkpoint_wait = sum(
        float(record.get("completion_wait_time", 0.0))
        for record in checkpoint_timings
    )
    checkpoint_save = sum(
        float(record.get("save_call_time", 0.0))
        for record in checkpoint_timings
    )
    checkpoint_count = len(checkpoint_timings)
    lines = [
        "# LaughLM PMAP Performance Baseline",
        "",
        "This report is generated from saved `metrics.jsonl` data. It does not",
        "execute JAX, model code, or accelerator runtime locally.",
        "",
        "## Run identity",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Selected steps: `{summary['first_step']} -> {summary['last_step']}`",
        f"- Rows: `{summary['rows_selected']} / {summary['rows_total']}`",
        f"- Skip steps: `{skip_steps}`",
        f"- Last N: `{last_n if last_n is not None else 'all'}`",
        *_manifest_summary(manifest),
        "",
        "## Steady-state metrics",
        "",
        f"- Median throughput: `{_fmt(summary.get('tokens_per_sec_median'), 1)} tokens/sec`",
        f"- Mean throughput: `{_fmt(summary.get('tokens_per_sec_mean'), 1)} tokens/sec`",
        f"- Median device throughput: `{_fmt(summary.get('device_tokens_per_sec_median'), 1)} tokens/sec`",
        f"- Median non-embedding MFU: `{_fmt(summary.get('mfu_non_embedding_median'), 2, '%')}`",
        f"- Median MFU with logits estimate: `{_fmt(summary.get('mfu_with_logits_estimate_median'), 2, '%')}`",
        f"- Loss: `{_fmt(summary.get('loss_first'), 5)} -> {_fmt(summary.get('loss_last'), 5)}`",
        f"- Inferred bottleneck: `{summary.get('bottleneck', 'unknown')}`",
        "",
        "## Timing breakdown",
        "",
        "| Component | Mean seconds | Share of mean step |",
        "|---|---:|---:|",
        f"| Total step | {_fmt(total_step)} | 100.0% |",
        f"| Data wait | {_fmt(summary.get('data_wait_time_mean'))} | {_pct(summary.get('data_wait_time_mean'), total_step)} |",
        f"| Host batch preparation | {_fmt(summary.get('host_batch_prepare_time_mean'))} | {_pct(summary.get('host_batch_prepare_time_mean'), total_step)} |",
        f"| Device transfer | {_fmt(summary.get('device_put_time_mean'))} | {_pct(summary.get('device_put_time_mean'), total_step)} |",
        f"| Device step | {_fmt(summary.get('device_step_time_mean'))} | {_pct(summary.get('device_step_time_mean'), total_step)} |",
        f"| Host overhead | {_fmt(summary.get('host_overhead_time_mean'))} | {_pct(summary.get('host_overhead_time_mean'), total_step)} |",
        "",
        "## Compilation and checkpoint notes",
        "",
        f"- First-step compile-plus-execute: `{_fmt(summary.get('first_step_compile_plus_execute_time'))} seconds`",
        "- Exact compile-only time: `pending TPU validation`; the first step includes execution.",
        "- Warm-cache reuse: compare reports whose manifests are marked `cold candidate` and `warm candidate`.",
        f"- Checkpoint records: `{checkpoint_count}`",
        f"- Checkpoint save-call time total: `{_fmt(checkpoint_save)} seconds`",
        f"- Checkpoint completion-wait time total: `{_fmt(checkpoint_wait)} seconds`",
        f"- Checkpoint overhead total: `{_fmt(checkpoint_total)} seconds`",
        "",
        "## Runtime shape",
        "",
        f"- Tokens per step: `{_fmt(summary.get('tokens_in_step'), 0)}`",
        f"- Sequence length: `{_fmt(summary.get('seq_len'), 0)}`",
        f"- Global batch: `{_fmt(summary.get('global_batch'), 0)}`",
        f"- Effective global batch: `{_fmt(summary.get('effective_global_batch'), 0)}`",
        f"- Gradient accumulation: `{_fmt(summary.get('gradient_accumulation'), 0)}`",
        f"- Devices: `{_fmt(summary.get('num_devices'), 0)}`",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown PMAP performance baseline report."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics.jsonl or its checkpoint directory.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown output path; defaults beside metrics.jsonl.",
    )
    parser.add_argument(
        "--skip-steps",
        type=int,
        default=5,
        help="Exclude initial steps from steady-state statistics.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=None,
        help="Use only the last N rows after skipping warmup steps.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_path = resolve_metrics_path(args.metrics)
    summary = summarize_metrics(
        metrics_path,
        skip_steps=args.skip_steps,
        last_n=args.last_n,
    )
    manifest = _load_json(metrics_path.parent / "run_manifest.json")
    checkpoint_timings = _load_jsonl(
        metrics_path.parent / "checkpoint_timings.jsonl"
    )
    output_path = Path(args.output).expanduser() if args.output else (
        metrics_path.parent / "performance_baseline.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_report(
            metrics_path=metrics_path,
            summary=summary,
            manifest=manifest,
            checkpoint_timings=checkpoint_timings,
            skip_steps=args.skip_steps,
            last_n=args.last_n,
        ),
        encoding="utf-8",
    )
    print(f"[baseline] report written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
