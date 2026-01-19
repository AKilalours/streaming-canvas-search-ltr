import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timed(section: str, timings: dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    timings[section] = (time.perf_counter() - t0) * 1000.0
