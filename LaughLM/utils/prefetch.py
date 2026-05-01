
import threading
import queue


def prefetch_to_device(iterator, size=8):
    """
    Simple async prefetch pipeline (CPU ONLY).

    Responsibilities:
    - Background loading
    - No reshaping
    - No device sharding
    - No batching logic

    All structure is handled in trainer.
    """

    q = queue.Queue(maxsize=size)
    stop_token = object()

    # ------------------------------------------------------------
    # Producer (CPU thread)
    # ------------------------------------------------------------
    def producer():
        exc = None
        try:
            for batch in iterator:
                q.put(batch)
        except Exception as e:
            exc = e
        finally:
            q.put((stop_token, exc))

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # ------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------
    while True:
        item = q.get()

        if isinstance(item, tuple) and item[0] is stop_token:
            _, exc = item
            if exc is not None:
                raise RuntimeError(f"Prefetch failed: {exc}") from exc
            break

        yield item
