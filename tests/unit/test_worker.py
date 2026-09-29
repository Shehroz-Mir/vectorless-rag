"""The ingestion worker (queue watching, dispatch, recovery), the per-user locks and the wiring."""
import logging
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr

from tests.fakes import (
    FakePageRenderer,
    FakeUserIndexProvider,
    InMemoryDocumentRepository,
    InMemoryFigureRepository,
    InMemoryFileStore,
)
from vectorless_rag.config import Settings
from vectorless_rag.models import Document, DocumentChanges, DocumentStatus, NewDocument
from vectorless_rag.operations import UserLocks
from vectorless_rag.worker import IngestionWorker, PerUserLocks, build_ingestion_worker

WAIT_S = 5.0


def add_queued(documents: InMemoryDocumentRepository, name: str = "manual.pdf") -> Document:
    return documents.add(NewDocument(
        id=uuid4(), user_id="alice", filename=name, original_path=Path(name), file_sha256=uuid4().hex * 2, page_count=1,
    ))


class RecordingIngest:
    """Stands in for DocumentIngestion.ingest: optionally waits for a gate, then marks the document done."""

    def __init__(self, documents: InMemoryDocumentRepository, gate: threading.Event | None = None) -> None:
        self.documents = documents
        self.gate = gate
        self.seen: list[UUID] = []
        self._done = threading.Semaphore(0)

    def __call__(self, document: Document) -> Document:
        if self.gate is not None:
            assert self.gate.wait(WAIT_S)
        self.seen.append(document.id)
        updated = self.documents.update(document.id, DocumentChanges(status=DocumentStatus.COMPLETED))
        self._done.release()
        return updated

    def wait_for(self, count: int) -> None:
        for _ in range(count):
            assert self._done.acquire(timeout=WAIT_S), "ingestion did not run"


def test_each_queued_document_is_dispatched_once_even_while_it_runs() -> None:
    documents = InMemoryDocumentRepository()
    first, second = add_queued(documents, "a.pdf"), add_queued(documents, "b.pdf")
    gate = threading.Event()
    ingest = RecordingIngest(documents, gate)
    worker = IngestionWorker(documents, ingest, max_parallel=2)

    started = worker.dispatch_queued()
    started_again = worker.dispatch_queued()  # both still queued in the registry, but already running
    gate.set()
    ingest.wait_for(2)
    worker.stop()

    assert (started, started_again) == (2, 0)
    assert sorted(ingest.seen) == sorted([first.id, second.id])


def test_start_requeues_interrupted_documents_and_runs_them() -> None:
    documents = InMemoryDocumentRepository()
    interrupted = add_queued(documents)
    documents.update(interrupted.id, DocumentChanges(status=DocumentStatus.INDEXING))
    ingest = RecordingIngest(documents)
    worker = IngestionWorker(documents, ingest, max_parallel=1, poll_interval_s=60)

    worker.start()
    ingest.wait_for(1)
    worker.stop()

    assert ingest.seen == [interrupted.id]


def test_wake_starts_a_new_upload_without_waiting_for_the_poll() -> None:
    documents = InMemoryDocumentRepository()
    ingest = RecordingIngest(documents)
    worker = IngestionWorker(documents, ingest, max_parallel=1, poll_interval_s=60)
    worker.start()

    uploaded = add_queued(documents)
    worker.wake()
    ingest.wait_for(1)
    worker.stop()

    assert ingest.seen == [uploaded.id]


def test_a_crash_is_logged_and_the_worker_keeps_going(caplog: pytest.LogCaptureFixture) -> None:
    documents = InMemoryDocumentRepository()
    broken, fine = add_queued(documents, "a.pdf"), add_queued(documents, "b.pdf")
    recording = RecordingIngest(documents)

    def ingest(document: Document) -> Document:
        if document.id == broken.id:
            raise RuntimeError("registry unreachable")
        return recording(document)

    worker = IngestionWorker(documents, ingest, max_parallel=1)

    with caplog.at_level(logging.ERROR):
        worker.dispatch_queued()
        recording.wait_for(1)
        worker.stop()

    assert recording.seen == [fine.id]
    assert f"ingestion of document {broken.id} stopped unexpectedly" in caplog.text


def test_start_twice_is_an_error() -> None:
    worker = IngestionWorker(InMemoryDocumentRepository(), lambda _document: None, max_parallel=1, poll_interval_s=60)
    worker.start()
    try:
        with pytest.raises(RuntimeError, match="already running"):
            worker.start()
    finally:
        worker.stop()


def test_user_locks_are_per_user() -> None:
    locks = PerUserLocks()
    acquired: list[bool] = []

    def try_lock(user_key: str) -> None:
        lock = locks.for_user(user_key)
        acquired.append(lock.acquire(timeout=0.1))
        if acquired[-1]:
            lock.release()

    with locks.for_user("alice"):
        for user_key in ("alice", "bob"):
            thread = threading.Thread(target=try_lock, args=(user_key,))
            thread.start()
            thread.join()

    assert acquired == [False, True]  # alice's lock is taken, bob's is free
    port: UserLocks = locks
    assert port.for_user("alice") is locks.for_user("alice")


def test_build_ingestion_worker_wires_the_use_case() -> None:
    documents = InMemoryDocumentRepository()
    worker = build_ingestion_worker(
        Settings(openai_api_key=SecretStr("sk-test")),
        documents=documents,
        figures=InMemoryFigureRepository(),
        files=InMemoryFileStore(),
        renderer=FakePageRenderer(),
        indexes=FakeUserIndexProvider(),
        locks=PerUserLocks(),
    )

    assert isinstance(worker, IngestionWorker)
    worker.stop()
