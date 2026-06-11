#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from LaughLM.analysis.metrics import (
    print_metrics_summary,
    summarize_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize LaughLM metrics.jsonl throughput/timing/MFU."
    )

    parser.add_argument(
        "--metrics",
        "--input",
        dest="metrics",
        required=True,
        help="Path to metrics.jsonl or run/checkpoint directory containing metrics.jsonl.",
    )

    parser.add_argument(
        "--skip_steps",
        type=int,
        default=0,
        help="Ignore rows with step <= skip_steps.",
    )

    parser.add_argument(
        "--last_n",
        type=int,
        default=None,
        help="Summarize only the last N rows after skip_steps.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON summary instead of human-readable text.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    summary = summarize_metrics(
        args.metrics,
        skip_steps=args.skip_steps,
        last_n=args.last_n,
    )

    if args.json:
        print(
            json.dumps(
                summary,
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print_metrics_summary(
            summary
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
