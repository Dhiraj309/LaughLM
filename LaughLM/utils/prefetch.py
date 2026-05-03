import collections
import jax


def prefetch_to_device(iterator, size=2):
    """
    Prefetch batches to device.

    Keeps `size` batches queued on device so TPU never waits
    for host batch preparation.

    Parameters
    ----------
    iterator : Python iterator yielding batches
    size : number of batches to prefetch

    Returns
    -------
    generator yielding device arrays
    """

    queue = collections.deque()

    def _device_put(batch):
        return jax.device_put(batch)

    # Fill queue
    for _ in range(size):
        batch = next(iterator)
        queue.append(_device_put(batch))

    while True:
        yield queue.popleft()

        batch = next(iterator)
        queue.append(_device_put(batch))
