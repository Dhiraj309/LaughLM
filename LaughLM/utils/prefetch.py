
"""
LaughLM/utils/prefetch.py

Async prefetch pipeline — CPU-side buffering only.

Responsibilities:
- Background loading of numpy batches from the data iterator
- Queued ahead-of-time so the training loop never waits for data
- NO device transfer here — trainer.py handles reshaping + device_put

FIX (frontier-optim audit 2026):
  The old code transferred data to jax.devices()[0] in the prefetch
  thread. This is WRONG for pmap training where data must be shaped
  (num_devices, micro_batch_per_device, seq_len) before device_put.
  Transferring to device[0] means only one device gets the data,
  breaking multi-device training.

  The correct pattern (following MaxText):
  - Prefetch thread: CPU-side buffering only (numpy arrays in queue)
  - Training loop: reshape to (devices, ...) then jax.device_put or
    let pmap handle distribution automatically

  This separation ensures:
  - No premature device binding
  - No shape assumptions in the data pipeline
  - Clean boundary between data loading and device placement
"""

import threading
import queue
import numpy as np


def prefetch_to_device(iterator, size=8):
    """
    Async prefetch pipeline — CPU-side buffering.

    The producer thread:
    1. Gets next batch from the iterator (numpy)
    2. Ensures contiguous memory layout
    3. Puts the numpy batch into the queue

    The consumer (training loop) gets numpy batches directly
    from the queue, then handles reshaping + device placement.

    Parameters
    ----------
    iterator : iterable yielding numpy arrays
    size     : max number of prefetched batches (default 8)

    Yields
    ------
    numpy arrays (CPU-resident, contiguous)
    """
    q = queue.Queue(maxsize=size)
    stop_token = object()

    # ── Producer (CPU thread — no device transfer) ──
    def producer():
        exc = None
        try:
            for batch in iterator:
                # Ensure contiguous numpy array for efficient transfer later
                if not isinstance(batch, np.ndarray):
                    batch = np.asarray(batch)
                if not batch.flags['C_CONTIGUOUS']:
                    batch = np.ascontiguousarray(batch)
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

        if isinstance(item, tuple) and len(item) == 2 and item[0] is stop_token:
            _, exc = item
            if exc is not None:
                raise RuntimeError(f"Prefetch failed: {exc}") from exc
            break

        yield item
