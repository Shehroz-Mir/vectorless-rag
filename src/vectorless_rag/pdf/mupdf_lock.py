"""One thread at a time in PyMuPDF.

PyMuPDF "does not support running on multiple threads - doing so may cause incorrect behaviour or
even crash Python itself" (PyMuPDF docs, Multiprocessing). The ingestion worker and the API both run
pdf/ code on threads, so every pdf/ entry point holds this process-wide lock while it runs.
Calls are short (a page render is ~0.1 s); detecting figures in a long document takes longest.
"""
from __future__ import annotations

import functools
import threading
from collections.abc import Callable

MUPDF_LOCK = threading.RLock()  # re-entrant, so an entry point may call another


def with_mupdf_lock[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(function)
    def locked(*args: P.args, **kwargs: P.kwargs) -> R:
        with MUPDF_LOCK:
            return function(*args, **kwargs)

    return locked
