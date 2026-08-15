#!/usr/bin/env python3
"""Compare saved MHA and GQA TPU artifacts without running model code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from LaughLM.analysis.metrics import resolve_metrics_path, summarize_metrics


def _load_manifest(metrics_path: Path) -> dict[str, Any]:
    path = metrics_path.parent / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _resolved_config(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("resolved_config", {})
    return value if isinstance(value, dict) else {}


def _attention_label(manifest: dict[str, Any]) -> str:
    config = _resolved_config(manifest)
    model = config.get("model", {})
    architecture = config.get("architecture", {})
    if not isinstance(model, dict):
        model = {}
    if not isinstance(architecture, dict):
        architecture = {}
    return (
        f"{architecture.get('attention_variant', 'unknown')} "
        f"({model.get('num_heads', '?')}/"
        f"{model.get('num_kv_heads', '?')})"
    )


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def _cache_state(manifest: dict[str, Any]) -> str:
    cache = manifest.get("compilation_cache", {})
    if not isinstance(cache, dict):
        return "unknown"
    file_count = cache.get("file_count_before_run", 0)
    if not isinstance(file_count, int) or file_count < 0:
        return "unknown"
    return "warm candidate" if file_count > 0 else "cold candidate"


def build_report(
    *,
    mha_path: Path,
    gqa_path: Path,
    mha_summary: dict[str, Any],
    gqa_summary: dict[str, Any],
    mha_manifest: dict[str, Any],
    gqa_manifest: dict[str, Any],
    skip_steps: int,
    last_n: int | None,
) -> str:
    rows = [
        ("Median tokens/sec", "tokens_per_sec_median", 1),
        ("Median device tokens/sec", "device_tokens_per_sec_median", 1),
        ("Median non-embedding MFU (%)", "mfu_non_embedding_median", 2),
        (
            "First-step compile+execute (sec)",
            "first_step_compile_plus_execute_time",
            3,
        ),
        ("Mean total step time (sec)", "total_step_time_mean", 3),
        ("Mean device step time (sec)", "device_step_time_mean", 3),
        ("Final loss", "loss_last", 5),
    ]

    lines = [
        "# LaughLM MHA vs GQA Comparison",
        "",
        "This report reads saved metrics and manifests only. It does not execute",
        "JAX, model code, or accelerator runtime locally.",
        "",
        "## Run identity",
        "",
        f"- MHA metrics: `{mha_path}`",
        f"- GQA metrics: `{gqa_path}`",
        f"- Selection: skip first `{skip_steps}` steps; last N `{last_n if last_n is not None else 'all'}`",
        f"- MHA attention: `{_attention_label(mha_manifest)}`",
        f"- GQA attention: `{_attention_label(gqa_manifest)}`",
        f"- MHA cache: `{_cache_state(mha_manifest)}`",
        f"- GQA cache: `{_cache_state(gqa_manifest)}`",
        "",
        "## Comparison",
        "",
        "| Metric | MHA | GQA | GQA - MHA |",
        "|---|---:|---:|---:|",
    ]

    for label, key, digits in rows:
        mha_value = mha_summary.get(key)
        gqa_value = gqa_summary.get(key)
        delta = (
            None
            if mha_value is None or gqa_value is None
            else float(gqa_value) - float(mha_value)
        )
        lines.append(
            f"| {label} | {_fmt(mha_value, digits)} | "
            f"{_fmt(gqa_value, digits)} | {_fmt(delta, digits)} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Accept a GQA change only when both runs use the same TPU shape,",
            "  shard selection, step window, and comparable cache state.",
            "- Confirm the GQA manifest and TPU logs show Splash dispatch with",
            "  no fallback before treating throughput or loss as comparable.",
            "- Memory outcome and fallback/error details require TPU logs or",
            "  profiler artifacts; they cannot be inferred from this table.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare saved MHA and GQA metrics/manifests."
    )
    parser.add_argument("--mha", required=True, help="MHA metrics path or run directory.")
    parser.add_argument("--gqa", required=True, help="GQA metrics path or run directory.")
    parser.add_argument("--output", default="attention_comparison.md")
    parser.add_argument("--skip-steps", type=int, default=5)
    parser.add_argument("--last-n", type=int, default=None)
    args = parser.parse_args()

    mha_path = resolve_metrics_path(args.mha)
    gqa_path = resolve_metrics_path(args.gqa)
    mha_manifest = _load_manifest(mha_path)
    gqa_manifest = _load_manifest(gqa_path)

    report = build_report(
        mha_path=mha_path,
        gqa_path=gqa_path,
        mha_summary=summarize_metrics(
            mha_path,
            skip_steps=args.skip_steps,
            last_n=args.last_n,
        ),
        gqa_summary=summarize_metrics(
            gqa_path,
            skip_steps=args.skip_steps,
            last_n=args.last_n,
        ),
        mha_manifest=mha_manifest,
        gqa_manifest=gqa_manifest,
        skip_steps=args.skip_steps,
        last_n=args.last_n,
    )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"[attention-compare] report written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
