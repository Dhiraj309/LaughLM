import threading
import queue
import jax


def prefetch_to_device(iterator, size=8):
    """
    High-performance prefetch pipeline for TPU/GPU.

    ✅ FIXES:
    - Larger queue (prevents pipeline starvation)
    - Proper async overlap
    - Exception propagation
    - Stable device sharding
    - Warmup-friendly behavior
    """

    # ✅ Bigger buffer = smoother pipeline
    q = queue.Queue(maxsize=size)

    stop_token = object()

    devices = jax.devices()
    n_devices = len(devices)

    # ------------------------------------------------------------
    # Producer thread (CPU → device transfer)
    # ------------------------------------------------------------
    def producer():
        exc = None
        try:
            for batch in iterator:

                # ✅ SAFETY: global batch must divide devices
                if batch.shape[0] % n_devices != 0:
                    raise ValueError(
                        f"Batch size {batch.shape[0]} not divisible by {n_devices} devices"
                    )

                per_device = batch.shape[0] // n_devices

                # (global_batch, seq) → (devices, per_device, seq)
                batch = batch.reshape(
                    n_devices,
                    per_device,
                    batch.shape[-1],
                )

                # ✅ CRITICAL: explicit list() ensures correct sharding
                device_batch = jax.device_put_sharded(
                    list(batch),
                    devices,
                )

                # ✅ NON-BLOCKING PUT (wait if queue full)
                q.put(device_batch)

        except Exception as e:
            exc = e

        finally:
            # ✅ propagate exception to main thread
            q.put((stop_token, exc))

    # ------------------------------------------------------------
    # Start background thread
    # ------------------------------------------------------------
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    # ------------------------------------------------------------
    # Consumer (training loop side)
    # ------------------------------------------------------------
    while True:
        item = q.get()

        # ✅ STOP HANDLING
        if isinstance(item, tuple) and item[0] is stop_token:
            _, exc = item
            if exc is not None:
                raise RuntimeError(f"Prefetch producer failed: {exc}") from exc
            break

        yield item