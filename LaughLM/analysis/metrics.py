from __future__ import annotations

import json
import statistics
import sys

from pathlib import Path
from typing import Any, Iterator, Mapping


def resolve_metrics_path(path: str | Path) -> Path:
    """
    Accept either:
    - direct path to metrics.jsonl
    - run directory containing metrics.jsonl
    """

    p = Path(path).expanduser().resolve()

    if p.is_dir():
        candidate = p / "metrics.jsonl"

        if candidate.exists():
            return candidate

        raise FileNotFoundError(
            f"No metrics.jsonl found in directory: {p}"
        )

    if not p.exists():
        raise FileNotFoundError(
            f"Metrics file does not exist: {p}"
        )

    return p


def iter_metrics(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Stream metrics rows from JSONL.

    Malformed trailing lines are skipped automatically.
    """

    metrics_path = resolve_metrics_path(path)

    with metrics_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()

            if not raw:
                continue

            try:
                record = json.loads(raw)

            except json.JSONDecodeError as exc:
                print(
                    f"[metrics] skipping malformed line "
                    f"{line_no} in {metrics_path.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            if not isinstance(record, Mapping):
                print(
                    f"[metrics] skipping non-object line "
                    f"{line_no} in {metrics_path.name}",
                    file=sys.stderr,
                )
                continue

            yield dict(record)


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_metrics(path))


def _as_float(record: Mapping[str, Any], key: str) -> float | None:
    value = record.get(key)

    if value is None:
        return None

    try:
        return float(value)

    except Exception:
        return None


def _values(
    rows: list[dict[str, Any]],
    key: str,
) -> list[float]:
    out = []

    for row in rows:
        value = _as_float(row, key)

        if value is not None:
            out.append(value)

    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None

    return float(statistics.median(values))


def _last(values: list[float]) -> float | None:
    if not values:
        return None

    return float(values[-1])


def _first_nonzero(
    rows: list[dict[str, Any]],
    key: str,
) -> float | None:
    """Return the first positive timing value in a metric stream."""
    for row in rows:
        value = _as_float(row, key)
        if value is not None and value > 0.0:
            return value
    return None


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    skip_steps: int,
    last_n: int | None,
) -> list[dict[str, Any]]:
    selected = []

    for row in rows:
        step = row.get("step", 0)

        try:
            step = int(step)

        except Exception:
            step = 0

        if step <= skip_steps:
            continue

        selected.append(row)

    if last_n is not None and last_n > 0:
        selected = selected[-last_n:]

    return selected


def _infer_bottleneck(summary: Mapping[str, Any]) -> str:
    """
    Simple bottleneck heuristic.

    Uses mean timing fields when available.
    """

    timing = {
        "data_wait": summary.get("data_wait_time_mean"),
        "host_batch_prepare": summary.get("host_batch_prepare_time_mean"),
        "device_put": summary.get("device_put_time_mean"),
        "device_step": summary.get("device_step_time_mean"),
        "host_overhead": summary.get("host_overhead_time_mean"),
    }

    timing = {
        key: float(value)
        for key, value in timing.items()
        if value is not None
    }

    if not timing:
        return "unknown"

    total = sum(
        max(0.0, value)
        for value in timing.values()
    )

    if total <= 0.0:
        return "unknown"

    ranked = sorted(
        timing.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_name, top_value = ranked[0]

    top_share = top_value / total

    if top_share >= 0.45:
        return top_name

    return "mixed"


def summarize_metrics(
    path: str | Path,
    *,
    skip_steps: int = 0,
    last_n: int | None = None,
) -> dict[str, Any]:
    """
    Summarize steady-state training metrics.

    Parameters
    ----------
    path:
        metrics.jsonl or checkpoint/run directory containing metrics.jsonl.

    skip_steps:
        Ignore rows with step <= skip_steps.

    last_n:
        If provided, summarize only the last N rows after skip_steps.

    Returns
    -------
    dict
        Summary of throughput, timing, MFU, and likely bottleneck.
    """

    all_rows = load_metrics(path)

    rows = _select_rows(
        all_rows,
        skip_steps=skip_steps,
        last_n=last_n,
    )

    if not rows:
        raise ValueError(
            "No metrics rows selected.\n"
            f"  total_rows={len(all_rows)}\n"
            f"  skip_steps={skip_steps}\n"
            f"  last_n={last_n}"
        )

    summary = {
        "rows_total": int(len(all_rows)),
        "rows_selected": int(len(rows)),
        "first_step": int(rows[0].get("step", 0)),
        "last_step": int(rows[-1].get("step", 0)),

        # Throughput.
        "tokens_per_sec_mean": _mean(_values(rows, "tokens_per_sec")),
        "tokens_per_sec_median": _median(_values(rows, "tokens_per_sec")),
        "device_tokens_per_sec_mean": _mean(_values(rows, "device_tokens_per_sec")),
        "device_tokens_per_sec_median": _median(_values(rows, "device_tokens_per_sec")),

        # Timing.
        "total_step_time_mean": _mean(_values(rows, "total_step_time")),
        "device_step_time_mean": _mean(_values(rows, "device_step_time")),
        "data_wait_time_mean": _mean(_values(rows, "data_wait_time")),
        "host_batch_prepare_time_mean": _mean(_values(rows, "host_batch_prepare_time")),
        "device_put_time_mean": _mean(_values(rows, "device_put_time")),
        "input_pipeline_time_mean": _mean(_values(rows, "input_pipeline_time")),
        "host_overhead_time_mean": _mean(_values(rows, "host_overhead_time")),
        "first_step_compile_plus_execute_time": _first_nonzero(
            all_rows,
            "first_step_compile_plus_execute_time",
        ),

        # Raw sync/debug timing.
        "raw_sync_step_time_mean": _mean(_values(rows, "raw_sync_step_time")),
        "raw_device_step_time_mean": _mean(_values(rows, "raw_device_step_time")),
        "interval_steps_median": _median(_values(rows, "interval_steps")),
        "interval_wall_time_mean": _mean(_values(rows, "interval_wall_time")),

        # MFU.
        "mfu_non_embedding_median": _median(_values(rows, "mfu_non_embedding")),
        "mfu_with_logits_estimate_median": _median(_values(rows, "mfu_with_logits_estimate")),
        "mfu_e2e_non_embedding_median": _median(_values(rows, "mfu_e2e_non_embedding")),
        "mfu_e2e_with_logits_estimate_median": _median(_values(rows, "mfu_e2e_with_logits_estimate")),

        # Loss.
        "loss_first": _as_float(rows[0], "loss"),
        "loss_last": _as_float(rows[-1], "loss"),
        "ppl_last": _as_float(rows[-1], "ppl"),

        # Runtime shape.
        "tokens_in_step": _last(_values(rows, "tokens_in_step")),
        "seq_len": _last(_values(rows, "seq_len")),
        "global_batch": _last(_values(rows, "global_batch")),
        "micro_global_batch": _last(_values(rows, "micro_global_batch")),
        "effective_global_batch": _last(_values(rows, "effective_global_batch")),
        "gradient_accumulation": _last(_values(rows, "gradient_accumulation")),
        "num_devices": _last(_values(rows, "num_devices")),
        "benchmark_mode": bool(rows[-1].get("benchmark_mode", False)),
    }

    summary["bottleneck"] = _infer_bottleneck(summary)

    return summary


def print_metrics_summary(
    summary: Mapping[str, Any],
) -> None:
    """
    Human-readable summary for CLI/notebooks.
    """

    def fmt(value, suffix=""):
        if value is None:
            return "n/a"

        if isinstance(value, float):
            return f"{value:,.4f}{suffix}"

        return f"{value}{suffix}"

    print("[metrics] summary")
    print(f"  rows:              {summary['rows_selected']} / {summary['rows_total']}")
    print(f"  steps:             {summary['first_step']} -> {summary['last_step']}")
    print(f"  bottleneck:        {summary['bottleneck']}")
    print()
    print(f"  tok/s mean:        {fmt(summary['tokens_per_sec_mean'])}")
    print(f"  tok/s median:      {fmt(summary['tokens_per_sec_median'])}")
    print(f"  device tok/s mean: {fmt(summary['device_tokens_per_sec_mean'])}")
    print()
    print(f"  total step mean:   {fmt(summary['total_step_time_mean'], 's')}")
    print(f"  device step mean:  {fmt(summary['device_step_time_mean'], 's')}")
    print(f"  data wait mean:    {fmt(summary['data_wait_time_mean'], 's')}")
    print(f"  host prep mean:    {fmt(summary['host_batch_prepare_time_mean'], 's')}")
    print(f"  device put mean:   {fmt(summary['device_put_time_mean'], 's')}")
    print(f"  host overhead mean:{fmt(summary['host_overhead_time_mean'], 's')}")
    print(
        "  first step compile+execute: "
        f"{fmt(summary['first_step_compile_plus_execute_time'], 's')}"
    )
    print()
    print(f"  MFU median:        {fmt(summary['mfu_non_embedding_median'], '%')}")
    print(f"  MFU+logits median: {fmt(summary['mfu_with_logits_estimate_median'], '%')}")
