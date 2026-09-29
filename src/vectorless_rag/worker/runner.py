"""Ingestion worker (spec 5.4): runs the ingestion use case for queued documents; a composition root.

v1 runs inside the service process: a small thread pool, woken after each upload and polling the
registry as a backstop. The registry is the queue, so nothing is lost when the process stops.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from uuid import UUID

from vectorless_rag.config import Settings
from vectorless_rag.models import Document, DocumentStatus
from vectorless_rag.operations import (
    DocumentIngestion,
    DocumentRepository,
    FigureRepository,
    FileStore,
    IngestionRules,
    PageRenderer,
    UserIndexProvider,
    UserLocks,
    requeue_interrupted,
)
from vectorless_rag.pdf import DetectionRules, detect_figure_pages, read_page_texts, write_invisible_notes
from vectorless_rag.vision import OpenAiFigureDescriber

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5.0  # backstop only: an upload wakes the worker at once


class IngestionWorker:
    def __init__(
        self,
        documents: DocumentRepository,
        ingest: Callable[[Document], object],
        *,
        max_parallel: int,
        poll_interval_s: float = POLL_INTERVAL_S,
    ) -> None:
        self._documents = documents
        self._ingest = ingest
        self._poll_interval_s = poll_interval_s
        self._pool = ThreadPoolExecutor(max_workers=max_parallel, thread_name_prefix="ingestion")
        self._running: set[UUID] = set()  # handed to the pool and not finished yet
        self._running_lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._watcher: threading.Thread | None = None

    def start(self) -> None:
        """Re-queue what a previous run left unfinished, then start watching the queue."""
        if self._watcher is not None:
            raise RuntimeError("the ingestion worker is already running")
        requeue_interrupted(self._documents)
        self._watcher = threading.Thread(target=self._watch, name="ingestion-queue", daemon=True)
        self._watcher.start()

    def wake(self) -> None:
        """Called after an upload, so the new document starts without waiting for the next poll."""
        self._wake.set()

    def stop(self, wait: bool = True) -> None:
        """Take no more documents; with `wait`, let running ones finish. Queued ones stay queued."""
        self._stopping.set()
        self._wake.set()
        if self._watcher is not None:
            self._watcher.join()
        self._pool.shutdown(wait=wait, cancel_futures=True)

    def dispatch_queued(self) -> int:
        """Hand each queued document that is not already running to the pool. Returns how many started."""
        started = 0
        for document in self._documents.list_by_status(DocumentStatus.QUEUED):
            with self._running_lock:
                if document.id in self._running:
                    continue
                self._running.add(document.id)
            self._pool.submit(self._run, document)
            started += 1
        return started

    def _watch(self) -> None:
        while not self._stopping.is_set():
            self._wake.clear()  # before reading the queue, so a wake during the read is not lost
            try:
                self.dispatch_queued()
            except Exception:  # a registry hiccup must not end the watcher; the next poll retries
                logger.exception("could not read the ingestion queue")
            self._wake.wait(self._poll_interval_s)

    def _run(self, document: Document) -> None:
        try:
            self._ingest(document)
        except Exception:  # the use case stores failures itself; this is a failing registry
            logger.exception("ingestion of document %s stopped unexpectedly", document.id)
        finally:
            with self._running_lock:
                self._running.discard(document.id)


def build_ingestion_worker(
    settings: Settings,
    *,
    documents: DocumentRepository,
    figures: FigureRepository,
    files: FileStore,
    renderer: PageRenderer,
    indexes: UserIndexProvider,
    locks: UserLocks,
) -> IngestionWorker:
    """Wire the ingestion use case to the PDF steps and the OpenAI describer. The registry, file store,
    renderer, PageIndex pool and locks are shared with the API, so its composition root passes them in."""
    detection = DetectionRules(
        min_image_area_ratio=settings.min_image_area_ratio,
        min_graphic_cluster_ratio=settings.min_graphic_cluster_ratio,
        max_cluster_text_density=settings.max_cluster_text_density,
        min_vector_figure_area=settings.min_vector_figure_area,
    )
    ingestion = DocumentIngestion(
        documents=documents,
        figures=figures,
        files=files,
        read_page_texts=read_page_texts,
        detect_figure_pages=partial(detect_figure_pages, rules=detection),
        renderer=renderer,
        describer=OpenAiFigureDescriber(settings.openai_api_key.get_secret_value(), settings.vision_model),
        write_invisible_notes=write_invisible_notes,
        indexes=indexes,
        locks=locks,
        rules=IngestionRules(
            max_figure_pages=settings.max_figure_pages,
            vision_concurrency=settings.vision_concurrency,
            scanned_max_text_chars=settings.scanned_max_text_chars,
            scanned_page_share=settings.scanned_page_share,
        ),
    )
    return IngestionWorker(documents, ingestion.ingest, max_parallel=settings.ingestion_workers)
