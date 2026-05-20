from __future__ import annotations

import json
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
