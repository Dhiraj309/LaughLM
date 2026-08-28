#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

# ============================================================
# IMPORTANT:
# Force non-interactive backend BEFORE pyplot import
# ============================================================

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

try:
    import scienceplots  # noqa: F401
    _HAS_SCIENCEPLOTS = True
except Exception:
    _HAS_SCIENCEPLOTS = False

from LaughLM.analysis.metrics import load_metrics


# ============================================================
# Style
# ============================================================

if _HAS_SCIENCEPLOTS:
    plt.style.use(["science", "no-latex", "dark_background"])
else:
    plt.style.use("dark_background")

plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "lines.linewidth": 2.0,
    }
)

EMA_ALPHA = 0.15


# ============================================================
# Helpers
# ============================================================

def ema(series: pd.Series, alpha: float = EMA_ALPHA) -> pd.Series:
    return pd.Series(series).reset_index(drop=True).ewm(alpha=alpha).mean()


def _first_existing_column(df: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def xlabel_from_column(col: str) -> str:
    if col == "step":
        return "Step"
    if col in ("tokens_processed", "tokens_seen"):
        return "Tokens Processed"
    if col in ("wall_time", "timestamp"):
        return "Wall Time"
    return col.replace("_", " ").title()


# ============================================================
# Normalization
# ============================================================

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "tokens_processed" not in out.columns and "tokens_seen" in out.columns:
        out["tokens_processed"] = out["tokens_seen"]

    if "tokens_seen" not in out.columns and "tokens_processed" in out.columns:
        out["tokens_seen"] = out["tokens_processed"]

    if "wall_time" not in out.columns and "timestamp" in out.columns:
        out["wall_time"] = out["timestamp"]

    if "timestamp" not in out.columns and "wall_time" in out.columns:
        out["timestamp"] = out["wall_time"]

    numeric_cols = [
        "step",
        "loss",
        "ppl",
        "grad_norm",
        "learning_rate",
        "tokens_per_sec",
        "mfu",
        "step_time",
        "tokens_processed",
        "tokens_seen",
        "wall_time",
        "timestamp",
    ]

    out = _coerce_numeric(out, numeric_cols)

    if "step" in out.columns:
        out = (
            out.sort_values("step")
            .drop_duplicates(subset=["step"], keep="last")
            .reset_index(drop=True)
        )

    return out


# ============================================================
# Figure finalizer
# ============================================================

def _finalize_figure(fig, save_path: Path | None = None):
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    fig.canvas.draw()
    plt.close(fig)


# ============================================================
# Plotting
# ============================================================

def frontier_plot(
    x,
    y,
    title: str,
    ylabel: str,
    *,
    xlabel: str = "Step",
    logy: bool = False,
    smooth: bool = True,
    save_dir: Path | None = None,
):
    fig, ax = plt.subplots()

    ax.plot(x, y, alpha=0.25, linewidth=1.2, label="raw")

    if smooth:
        ax.plot(x, ema(y), linewidth=2.5, label="ema")

    ax.set_title(title, pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logy:
        ax.set_yscale("log")

    ax.legend(frameon=False)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        filename = title.lower().replace(" ", "_").replace("/", "_")
        save_path = save_dir / f"{filename}.png"

    _finalize_figure(fig, save_path)


def plot_dashboard(df: pd.DataFrame, *, x_col: str, save_dir: Path | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(df[x_col], df["loss"], alpha=0.25)
    axes[0, 0].plot(df[x_col], ema(df["loss"]))
    axes[0, 0].set_title("Loss")

    axes[0, 1].plot(df[x_col], df["mfu"], alpha=0.25)
    axes[0, 1].plot(df[x_col], ema(df["mfu"]))
    axes[0, 1].set_title("MFU")

    axes[1, 0].plot(df[x_col], df["tokens_per_sec"], alpha=0.25)
    axes[1, 0].plot(df[x_col], ema(df["tokens_per_sec"]))
    axes[1, 0].set_title("Tokens / Second")

    axes[1, 1].plot(df[x_col], df["grad_norm"], alpha=0.25)
    axes[1, 1].plot(df[x_col], ema(df["grad_norm"]))
    axes[1, 1].set_title("Gradient Norm")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(xlabel_from_column(x_col))

    plt.suptitle("LaughLM Training Dashboard", fontsize=18, y=1.02)
    plt.tight_layout()

    save_path = None
    if save_dir is not None:
        save_path = save_dir / "dashboard.png"

    _finalize_figure(fig, save_path)


# ============================================================
# Summary
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Loaded metrics")
    print("=" * 60)

    print(f"Rows: {len(df):,}")
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())

    if "step" in df.columns and df["step"].notna().any():
        print("\nStep range:")
        print(f"{int(df['step'].min()):,} -> {int(df['step'].max()):,}")

    if "loss" in df.columns and df["loss"].notna().any():
        print("\nBest loss:")
        print(f"{df['loss'].min():.6f}")


# ============================================================
# Notebook-safe display helper
# ============================================================

def maybe_display_dashboard(save_dir: Path, enable: bool):
    if not enable:
        return

    try:
        from IPython.display import Image, display

        path = save_dir / "dashboard.png"
        if path.exists():
            display(Image(str(path)))
    except Exception:
        pass


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Plot LaughLM metrics.")
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Path to metrics.jsonl or to a run directory containing "
            "metrics.jsonl. Do not pass a YAML training config."
        ),
    )
    parser.add_argument("--save-dir", type=str, default="plots")
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display dashboard inline (notebook only).",
    )
    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    save_dir = Path(args.save_dir).expanduser().resolve()

    # A YAML file is a model/training configuration, not a metrics stream.
    # Reject it before load_metrics() attempts to parse every YAML line as JSON.
    if input_path.is_file() and input_path.suffix.lower() in {".yaml", ".yml"}:
        raise ValueError(
            "--input must point to metrics.jsonl (or its run directory), "
            f"not a YAML config: {input_path}\n"
            "Example: --input checkpoints/testing/"
            "laughlm_v1_135m_8shards/metrics.jsonl"
        )

    rows = load_metrics(input_path)
    if not rows:
        raise ValueError(f"No valid metrics rows found in: {input_path}")

    df = normalize_dataframe(pd.DataFrame(rows))

    if df.empty:
        raise ValueError("Metrics dataframe is empty.")
    if "step" not in df.columns:
        raise ValueError("Missing 'step' column.")

    print_summary(df)

    x_col = "step"

    loss_col = _first_existing_column(df, "loss")
    ppl_col = _first_existing_column(df, "ppl")
    lr_col = _first_existing_column(df, "learning_rate")
    tps_col = _first_existing_column(df, "tokens_per_sec")
    mfu_col = _first_existing_column(df, "mfu")
    gnorm_col = _first_existing_column(df, "grad_norm")
    step_time_col = _first_existing_column(df, "step_time")
    tokens_col = _first_existing_column(df, "tokens_processed", "tokens_seen")

    if loss_col:
        frontier_plot(df[x_col], df[loss_col], "Training Loss", "Loss", save_dir=save_dir)
        frontier_plot(df[x_col], df[loss_col], "Training Loss Log Scale", "Loss", logy=True, save_dir=save_dir)

    if ppl_col:
        frontier_plot(df[x_col], df[ppl_col], "Perplexity", "PPL", logy=True, save_dir=save_dir)

    if lr_col:
        frontier_plot(df[x_col], df[lr_col], "Learning Rate", "LR", smooth=False, save_dir=save_dir)

    if tps_col:
        frontier_plot(df[x_col], df[tps_col], "Training Throughput", "Tokens / Second", save_dir=save_dir)

    if mfu_col:
        frontier_plot(df[x_col], df[mfu_col], "MFU", "MFU (%)", save_dir=save_dir)

    if gnorm_col:
        frontier_plot(df[x_col], df[gnorm_col], "Gradient Norm", "Norm", save_dir=save_dir)

    if step_time_col:
        frontier_plot(df[x_col], df[step_time_col], "Step Time", "Seconds", save_dir=save_dir)

    if tokens_col:
        frontier_plot(df[x_col], df[tokens_col], "Tokens Processed", "Tokens", smooth=False, save_dir=save_dir)

    dashboard_cols = [loss_col, mfu_col, tps_col, gnorm_col]

    if all(col is not None for col in dashboard_cols):
        plot_dashboard(df, x_col=x_col, save_dir=save_dir)

    print("\nSaved plots to:")
    print(save_dir)

    maybe_display_dashboard(save_dir, args.display)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
