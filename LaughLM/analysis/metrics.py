from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SERIES: tuple[tuple[str, str], ...] = (
    ("loss", "Loss"),
    ("ppl", "Perplexity"),
    ("grad_norm", "Gradient Norm"),
    ("learning_rate", "Learning Rate"),
    ("tokens_per_sec", "Tokens / Second"),
    ("mfu", "MFU (%)"),
    ("step_time", "Step Time (s)"),
)

X_AXIS_ALIASES: dict[str, tuple[str, ...]] = {
    "step": ("step",),
    "tokens_seen": ("tokens_seen", "tokens_processed"),
    "tokens_processed": ("tokens_processed", "tokens_seen"),
    "wall_time": ("wall_time", "timestamp"),
    "timestamp": ("timestamp", "wall_time"),
}


def _resolve_metrics_path(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    if p.is_dir():
        candidate = p / "metrics.jsonl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"No metrics.jsonl found in directory: {p}"
        )
    if not p.exists():
        raise FileNotFoundError(f"Metrics file does not exist: {p}")
    return p


def _coerce_float(value: Any) -> float:
    if value is None:
        return float("nan")

    if isinstance(value, bool):
        return float("nan")

    try:
        return float(value)
    except Exception:
        return float("nan")


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    tokens_seen = record.get("tokens_seen", record.get("tokens_processed"))
    wall_time = record.get("wall_time", record.get("timestamp"))

    normalized = {
        "step": _coerce_int(record.get("step")),
        "loss": _coerce_float(record.get("loss")),
        "ppl": _coerce_float(record.get("ppl")),
        "grad_norm": _coerce_float(record.get("grad_norm")),
        "learning_rate": _coerce_float(record.get("learning_rate")),
        "tokens_per_sec": _coerce_float(record.get("tokens_per_sec")),
        "mfu": _coerce_float(record.get("mfu")),
        "step_time": _coerce_float(record.get("step_time")),
        "tokens_seen": _coerce_float(tokens_seen),
        "tokens_processed": _coerce_float(
            record.get("tokens_processed", tokens_seen)
        ),
        "wall_time": _coerce_float(wall_time),
        "timestamp": _coerce_float(record.get("timestamp", wall_time)),
    }

    # Keep any extra keys around for downstream analysis/debugging.
    for key, value in record.items():
        if key not in normalized:
            normalized[key] = value

    return normalized


def iter_metrics(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Stream JSONL metrics records.

    Malformed lines are skipped so that interrupted writes do not break
    post-run plotting.
    """
    metrics_path = _resolve_metrics_path(path)

    with metrics_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue

            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(
                    f"[metrics] skipping malformed line {line_no} in "
                    f"{metrics_path.name}: {exc}",
                    file=sys.stderr,
                )
                continue

            if isinstance(record, Mapping):
                yield _normalize_record(record)
            else:
                print(
                    f"[metrics] skipping non-object line {line_no} in "
                    f"{metrics_path.name}",
                    file=sys.stderr,
                )


def load_metrics(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_metrics(path))


def smooth_ema(values: Sequence[float], alpha: float) -> np.ndarray:
    """
    Exponential moving average smoothing.

    alpha in (0, 1] controls how much of the new point is retained.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    if alpha <= 0.0 or alpha >= 1.0:
        return arr

    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, arr.size):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _axis_label(axis: str) -> str:
    if axis == "step":
        return "Step"
    if axis in ("tokens_seen", "tokens_processed"):
        return "Tokens Seen"
    if axis in ("wall_time", "timestamp"):
        return "Wall Time (s)"
    return axis.replace("_", " ").title()


def _series_from_records(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> np.ndarray:
    aliases = X_AXIS_ALIASES.get(field, (field,))
    values: list[float] = []

    for record in records:
        value = float("nan")
        for key in aliases:
            if key in record:
                value = _coerce_float(record.get(key))
                if math.isfinite(value):
                    break
        values.append(value)

    return np.asarray(values, dtype=np.float64)


def _finite_xy(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    return x[order], y[order]


def _plot_single_series(
    *,
    records: Sequence[Mapping[str, Any]],
    x_axis: str,
    y_field: str,
    y_label: str,
    output_path: Path,
    smooth_alpha: float = 0.0,
) -> bool:
    x = _series_from_records(records, x_axis)
    y = _series_from_records(records, y_field)
    x, y = _finite_xy(x, y)

    if x.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    ax.margins(x=0.02)

    if smooth_alpha and 0.0 < smooth_alpha < 1.0 and y.size >= 2:
        ax.plot(
            x,
            y,
            linewidth=1.0,
            alpha=0.28,
            label="raw",
        )
        ax.plot(
            x,
            smooth_ema(y, smooth_alpha),
            linewidth=1.8,
            label=f"EMA(alpha={smooth_alpha:g})",
        )
        ax.legend(frameon=False)
    else:
        ax.plot(x, y, linewidth=1.6)

    ax.set_title(f"{y_label} vs {_axis_label(x_axis)}")
    ax.set_xlabel(_axis_label(x_axis))
    ax.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def summarize_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        return {
            "num_records": 0,
        }

    step = _series_from_records(records, "step")
    loss = _series_from_records(records, "loss")
    ppl = _series_from_records(records, "ppl")
    grad_norm = _series_from_records(records, "grad_norm")
    lr = _series_from_records(records, "learning_rate")
    toks_per_sec = _series_from_records(records, "tokens_per_sec")
    mfu = _series_from_records(records, "mfu")
    step_time = _series_from_records(records, "step_time")
    tokens_seen = _series_from_records(records, "tokens_seen")
    wall_time = _series_from_records(records, "wall_time")

    summary: dict[str, Any] = {
        "num_records": int(len(records)),
    }

    if np.isfinite(step).any():
        summary["first_step"] = int(np.nanmin(step))
        summary["last_step"] = int(np.nanmax(step))

    if np.isfinite(loss).any():
        summary["best_loss"] = float(np.nanmin(loss))
        summary["final_loss"] = float(loss[np.isfinite(loss)][-1])

    if np.isfinite(ppl).any():
        summary["best_ppl"] = float(np.nanmin(ppl))
        summary["final_ppl"] = float(ppl[np.isfinite(ppl)][-1])

    if np.isfinite(grad_norm).any():
        summary["max_grad_norm"] = float(np.nanmax(grad_norm))
        summary["final_grad_norm"] = float(grad_norm[np.isfinite(grad_norm)][-1])

    if np.isfinite(lr).any():
        summary["final_learning_rate"] = float(lr[np.isfinite(lr)][-1])

    if np.isfinite(toks_per_sec).any():
        finite = toks_per_sec[np.isfinite(toks_per_sec)]
        summary["avg_tokens_per_sec"] = float(np.mean(finite))
        summary["final_tokens_per_sec"] = float(finite[-1])

    if np.isfinite(mfu).any():
        finite = mfu[np.isfinite(mfu)]
        summary["peak_mfu"] = float(np.max(finite))
        summary["final_mfu"] = float(finite[-1])

    if np.isfinite(step_time).any():
        finite = step_time[np.isfinite(step_time)]
        summary["avg_step_time"] = float(np.mean(finite))
        summary["final_step_time"] = float(finite[-1])

    if np.isfinite(tokens_seen).any():
        summary["final_tokens_seen"] = int(np.nanmax(tokens_seen))

    if np.isfinite(wall_time).any():
        finite = wall_time[np.isfinite(wall_time)]
        summary["start_wall_time"] = float(finite[0])
        summary["end_wall_time"] = float(finite[-1])
        summary["duration_seconds"] = float(finite[-1] - finite[0])

    return summary


def plot_metrics(
    metrics_path: str | Path,
    output_dir: str | Path,
    *,
    x_axis: str = "tokens_seen",
    series: Sequence[tuple[str, str]] = DEFAULT_SERIES,
    smooth_alpha: float = 0.0,
) -> dict[str, Any]:
    """
    Load metrics.jsonl and write plot PNGs plus a summary.json.

    Returns the summary dictionary.
    """
    metrics_path = _resolve_metrics_path(metrics_path)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_metrics(metrics_path)
    if not records:
        raise ValueError(f"No valid metrics records found in {metrics_path}")

    generated_plots: list[str] = []

    for field, label in series:
        out_path = output_dir / f"{field}_vs_{x_axis}.png"
        ok = _plot_single_series(
            records=records,
            x_axis=x_axis,
            y_field=field,
            y_label=label,
            output_path=out_path,
            smooth_alpha=smooth_alpha,
        )
        if ok:
            generated_plots.append(out_path.name)

    summary = summarize_metrics(records)
    summary.update(
        {
            "input_path": str(metrics_path),
            "output_dir": str(output_dir),
            "x_axis": x_axis,
            "generated_plots": generated_plots,
        }
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary
