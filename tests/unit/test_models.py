from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from vectorless_rag.models import (
    AgentRun,
    DetectedFigure,
    DocumentChanges,
    DocumentStatus,
    FigureKind,
    FigurePage,
    ModelStep,
    NewDocument,
    ToolOutcome,
    ToolStep,
)


def tool(index: int, name: str, document: str, *pages: int, outcome: ToolOutcome = ToolOutcome.OK) -> ToolStep:
    return ToolStep(index=index, tool=name, arguments={}, document=document, pages=pages, outcome=outcome, duration_ms=1)


def test_totals_count_unique_pages_per_document_and_every_image_viewed() -> None:
    run = AgentRun.of("Answer.", [
        ModelStep(index=1, duration_ms=10, input_tokens=100, output_tokens=5, reasoning_tokens=3),
        tool(2, "get_page_content", "a.pdf", 1, 2),
        tool(3, "view_pages", "a.pdf", 2),
        tool(4, "view_pages", "a.pdf", 2),
        tool(5, "get_page_content", "b.pdf", 2),
        tool(6, "view_pages", "b.pdf", outcome=ToolOutcome.BLOCKED),
        ModelStep(index=7, duration_ms=10, input_tokens=200, output_tokens=7),
    ], duration_ms=40)

    assert run.totals.model_dump() == {
        "model_calls": 2, "tool_calls": 5, "pages_read": 3, "images_viewed": 2,
        "input_tokens": 300, "output_tokens": 12, "reasoning_tokens": 3, "duration_ms": 40,
    }


def test_a_run_survives_json_with_its_step_kinds() -> None:
    run = AgentRun.of("A.", [ModelStep(index=1, duration_ms=1, tool_calls=("view_pages",)), tool(2, "view_pages", "a.pdf", 3)], duration_ms=2)

    again = AgentRun.model_validate_json(run.model_dump_json())

    assert again == run and [type(step) for step in again.steps] == [ModelStep, ToolStep]


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
