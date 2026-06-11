#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from LaughLM.analysis.metrics import summarize_metrics


def parse_run_arg(value: str) -> tuple[str, str]:
    """
    Parse:
      name=path
    """

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run must use format name=path"
        )

    name, path = value.split(
        "=",
        1,
    )

    name = name.strip()
    path = path.strip()

    if not name:
        raise argparse.ArgumentTypeError(
            "Run name cannot be empty"
        )

    if not path:
        raise argparse.ArgumentTypeError(
            "Run path cannot be empty"
        )

    return name, path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare LaughLM metrics summaries across runs."
    )

    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        help="Run in format name=metrics_path_or_run_dir. Can be repeated.",
    )

    parser.add_argument(
        "--skip_steps",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--last_n",
        type=int,
        default=None,
    )

    return parser.parse_args()


def _fmt_num(value, digits=2):
    if value is None:
        return "n/a"

    return f"{float(value):,.{digits}f}"


def _fmt_pct(value):
    if value is None:
        return "n/a"

    return f"{float(value):.2f}%"


def main() -> int:
    args = parse_args()

    rows = []

    for name, path in args.run:
        summary = summarize_metrics(
            Path(path),
            skip_steps=args.skip_steps,
            last_n=args.last_n,
        )

        rows.append(
            {
                "name": name,
                "path": path,
                "summary": summary,
            }
        )

    if not rows:
        raise ValueError(
            "No runs provided."
        )

    baseline_tps = rows[0]["summary"]["tokens_per_sec_median"]

    print()
    print("[compare_metrics]")
    print(f"  baseline:   {rows[0]['name']}")
    print(f"  skip_steps: {args.skip_steps}")
    print(f"  last_n:     {args.last_n}")
    print()

    header = (
        f"{'run':<28}"
        f"{'rows':>8} "
        f"{'tok/s med':>12} "
        f"{'speedup':>9} "
        f"{'dev step':>10} "
        f"{'MFU':>9} "
        f"{'MFU+logits':>12} "
        f"{'loss last':>11} "
        f"{'bottleneck':>14}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        s = row["summary"]

        tps = s["tokens_per_sec_median"]

        if baseline_tps is None or baseline_tps <= 0 or tps is None:
            speedup = None
        else:
            speedup = (
                (float(tps) / float(baseline_tps) - 1.0)
                * 100.0
            )

        print(
            f"{row['name']:<28}"
            f"{int(s['rows_selected']):>8} "
            f"{_fmt_num(tps):>12} "
            f"{_fmt_pct(speedup):>9} "
            f"{_fmt_num(s['device_step_time_mean'], 4):>10} "
            f"{_fmt_pct(s['mfu_non_embedding_median']):>9} "
            f"{_fmt_pct(s['mfu_with_logits_estimate_median']):>12} "
            f"{_fmt_num(s['loss_last'], 4):>11} "
            f"{str(s['bottleneck']):>14}"
        )

    print()
    print("Paths:")
    for row in rows:
        print(f"  {row['name']}: {row['path']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
