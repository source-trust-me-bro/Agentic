"""One way to run slow, independent work side by side.

Almost everything the agents wait for is a network call (a model, a news feed, a web page), so
threads are the right tool: while one call waits, the others run. Results come back in the
order the items went in, whichever finishes first, so output stays predictable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def run_parallel[T, R](
    fn: Callable[[T], R], items: Iterable[T], workers: int, name: str = "newsroom"
) -> list[R]:
    """``[fn(item) for item in items]``, a few at a time. ``workers`` of 1 runs them in turn."""

    items = list(items)
    workers = max(1, min(workers, len(items)))
    if workers == 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=name) as pool:
        return list(pool.map(fn, items))


def start_in_background(fn: Callable[[], Any], name: str = "newsroom") -> Callable[[], Any]:
    """Start ``fn`` now on its own thread. Call the returned function to wait for its result
    (and to re-raise anything it raised)."""

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
    future = pool.submit(fn)

    def wait() -> Any:
        try:
            return future.result()
        finally:
            pool.shutdown(wait=False)

    return wait
