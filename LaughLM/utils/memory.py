"""
LaughLM/utils/memory.py

Simple TPU/GPU memory instrumentation.
"""

from __future__ import annotations

import jax


def print_memory_stats(
    prefix: str = "",
):
    """
    Print per-device memory stats.
    """

    devices = jax.devices()

    for device in devices:

        try:

            stats = device.memory_stats()

        except Exception:

            continue

        used = stats.get(
            "bytes_in_use",
            0,
        )

        limit = stats.get(
            "bytes_limit",
            0,
        )

        peak = stats.get(
            "peak_bytes_in_use",
            0,
        )

        gb = 1024 ** 3

        print(
            f"{prefix}"
            f"{device}: "
            f"used={used / gb:.2f} GB "
            f"peak={peak / gb:.2f} GB "
            f"limit={limit / gb:.2f} GB",
            flush=True,
        )
