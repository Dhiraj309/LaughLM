
"""
LaughLM/utils/prefetch.py

Async prefetch pipeline with device-side transfer.

Responsibilities:
- Background loading of numpy batches from the data iterator
- Device transfer (numpy → JAX array) happens in the prefetch thread
  so the TPU/GPU never waits for CPU→device transfer
- No reshaping, no sharding, no batching logic

All data structure transformations are handled in trainer.py.
"""

import threading
import queue
import jax
import jax.numpy as jnp
import numpy as np


def prefetch_to_device(iterator, size=8):
    """
    Async prefetch pipeline with device-side transfer.

    The producer thread:
    1. Gets next batch from the iterator (numpy)
    2. Converts to JAX array and transfers to first device
    3. Puts the device-resident batch into the queue

    The consumer (training loop) gets device-resident batches
    directly from the queue — no blocking host→device transfer.

    Parameters
    ----------
    iterator : iterable yielding numpy arrays
    size     : max number of prefetched batches (default 8)

    Yields
    ------
    JAX arrays resident on the default device
    """
    q = queue.Queue(maxsize=size)
    stop_token = object()

    # ── Get first device for transfer hints ───
    try:
        device = jax.devices()[0]
    except Exception:
        device = None

    # ── Producer (CPU thread with device transfer) ──
    def producer():
        exc = None
        try:
            for batch in iterator:
                # Convert numpy → JAX and transfer to device
                if isinstance(batch, np.ndarray):
                    batch = jax.device_put(batch, device) if device else jnp.asarray(batch)
                elif isinstance(batch, jax.Array):
                    pass  # Already on device
                else:
                    batch = jax.device_put(np.asarray(batch), device) if device else jnp.asarray(batch)
                q.put(batch)
        except Exception as e:
            exc = e
        finally:
            q.put((stop_token, exc))

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # ── Consumer ──
    while True:
        item = q.get()

        if isinstance(item, tuple) and item[0] is stop_token:
            _, exc = item
            if exc is not None:
                raise RuntimeError(f"Prefetch failed: {exc}") from exc
            break

        yield item