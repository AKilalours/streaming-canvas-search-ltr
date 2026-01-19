import time
from contextlib import contextmanager
from typing import Dict, Iterator


@contextmanager
def timed(section: str, timings: Dict[str, float]) -> Iterator[None]:
    t0 = time.perf_counter()
    yield
    timings[section] = (time.perf_counter() - t0) * 1000.0

