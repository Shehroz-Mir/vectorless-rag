"""pyright proves each fake satisfies its Protocol: an assignment below fails type-checking if not."""
from pathlib import Path

from tests.fakes.files import InMemoryFileStore
from tests.fakes.registry import InMemoryDocumentRepository, InMemoryFigureRepository
from tests.fakes.services import (
    FakeAnswerAgent,
    FakeFigureDescriber,
    FakePageRenderer,
    FakeUserIndex,
    FakeUserIndexProvider,
)
from vectorless_rag.operations.ports import (
    AnswerAgent,
    DocumentRepository,
    FigureDescriber,
    FigureRepository,
    FileStore,
    PageRenderer,
    UserIndex,
    UserIndexProvider,
)


def test_fakes_satisfy_the_ports() -> None:
    documents: DocumentRepository = InMemoryDocumentRepository()
    figures: FigureRepository = InMemoryFigureRepository()
    files: FileStore = InMemoryFileStore()
    renderer: PageRenderer = FakePageRenderer()
    describer: FigureDescriber = FakeFigureDescriber()
    index: UserIndex = FakeUserIndex()
    provider: UserIndexProvider = FakeUserIndexProvider()
    agent: AnswerAgent = FakeAnswerAgent("answer")

    assert all(port is not None for port in (documents, figures, files, renderer, describer, index, provider, agent))


def test_fake_index_uniquifies_names_and_resolves_citations() -> None:
    index = FakeUserIndex()
    first = index.submit(Path("report.pdf"))
    second = index.submit(Path("other/report.pdf"))

    resolved = index.resolve_citations(
        'A <cite doc="report.pdf" page="3"/> B <cite doc="report_1.pdf" page="1"/> '
        'C <cite doc="report.pdf" page="3"/> D <cite doc="missing.pdf" page="2"/>'
    )

    assert second.name == "report_1.pdf"
    assert resolved.text == "A [1] B [2] C [1] D [3]"
    assert [(c.index, c.doc_id) for c in resolved.citations] == [(1, first.doc_id), (2, second.doc_id), (3, None)]
