"""In-memory versions of every port, for testing the use cases without a database, OpenAI or PageIndex."""
from tests.fakes.clock import TickingClock
from tests.fakes.files import InMemoryFileStore
from tests.fakes.locks import RecordingLocks
from tests.fakes.registry import InMemoryDocumentRepository, InMemoryFigureRepository
from tests.fakes.services import (
    FakeAnswerAgent,
    FakeFigureDescriber,
    FakePageRenderer,
    FakeUserIndex,
    FakeUserIndexProvider,
)

__all__ = [
    "FakeAnswerAgent",
    "FakeFigureDescriber",
    "FakePageRenderer",
    "FakeUserIndex",
    "FakeUserIndexProvider",
    "InMemoryDocumentRepository",
    "InMemoryFigureRepository",
    "InMemoryFileStore",
    "RecordingLocks",
    "TickingClock",
]
