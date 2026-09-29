from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vectorless_rag.models import (
    DetectedFigure,
    DocumentChanges,
    DocumentStatus,
    FigureKind,
    FigurePage,
    NewDocument,
)


def test_changes_apply_only_the_fields_that_were_set() -> None:
    changes = DocumentChanges(status=DocumentStatus.INDEXING)

    assert changes.as_update() == {"status": DocumentStatus.INDEXING}


def test_changes_can_clear_an_error_explicitly() -> None:
    changes = DocumentChanges(status=DocumentStatus.QUEUED, error=None)

    assert changes.as_update() == {"status": DocumentStatus.QUEUED, "error": None}


def test_main_figure_is_the_largest() -> None:
    page = FigurePage(
        page=14,
        figures=(
            DetectedFigure(kind=FigureKind.RASTER, box=(0, 0, 10, 10)),
            DetectedFigure(kind=FigureKind.VECTOR, box=(0, 0, 100, 50)),
        ),
    )

    assert page.main_figure.kind is FigureKind.VECTOR


def test_figure_page_needs_at_least_one_figure() -> None:
    with pytest.raises(ValidationError):
        FigurePage(page=1, figures=())


@pytest.mark.parametrize("sha256", ["abc", "G" * 64, "A" * 64])
def test_new_document_rejects_bad_hashes(sha256: str) -> None:
    with pytest.raises(ValidationError):
        NewDocument(id=uuid4(), user_id="u1", filename="a.pdf", original_path=Path("a.pdf"), file_sha256=sha256, page_count=1)
