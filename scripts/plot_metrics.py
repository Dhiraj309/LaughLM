#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from LaughLM.analysis.metrics import plot_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LaughLM JSONL training metrics after a run completes."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to metrics.jsonl or to the run directory containing it.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory where plots will be written. Defaults to <run_dir>/plots.",
    )
    parser.add_argument(
        "--x-axis",
        default="tokens_seen",
        choices=("step", "tokens_seen", "tokens_processed", "wall_time", "timestamp"),
        help="X axis to use for all plots.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.0,
        help="EMA smoothing alpha in (0, 1]. Use 0 to disable smoothing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    if args.output is None:
        if input_path.is_dir():
            output_dir = input_path / "plots"
        else:
            output_dir = input_path.parent / "plots"
    else:
        output_dir = Path(args.output).expanduser().resolve()

    summary = plot_metrics(
        metrics_path=input_path,
        output_dir=output_dir,
        x_axis=args.x_axis,
        smooth_alpha=args.smooth_alpha,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
