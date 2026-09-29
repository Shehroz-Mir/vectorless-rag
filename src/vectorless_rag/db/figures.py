"""The `figure_descriptions` table behind ports.FigureRepository."""
from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, select

from vectorless_rag.db.base import SqlRepository, as_utc
from vectorless_rag.db.tables import FigureRow
from vectorless_rag.models import FigureDescription, NewFigureDescription


class SqlFigureRepository(SqlRepository):
    def add_many(self, figures: Sequence[NewFigureDescription]) -> list[FigureDescription]:
        now = self._now()
        rows = [FigureRow(id=uuid4(), created_at=now, **figure.model_dump()) for figure in figures]
        with self._transaction() as session:
            session.add_all(rows)
        return [_to_figure(row) for row in rows]

    def list_for_document(self, document_id: UUID) -> list[FigureDescription]:
        statement = select(FigureRow).where(FigureRow.document_id == document_id)
        with self._transaction() as session:
            return [_to_figure(row) for row in session.scalars(statement.order_by(FigureRow.page, FigureRow.figure_index))]

    def delete_for_document(self, document_id: UUID) -> None:
        with self._transaction() as session:
            session.execute(delete(FigureRow).where(FigureRow.document_id == document_id))


def _to_figure(row: FigureRow) -> FigureDescription:
    return FigureDescription(
        id=row.id,
        document_id=row.document_id,
        page=row.page,
        figure_index=row.figure_index,
        kind=row.kind,
        description=row.description,
        vision_model=row.vision_model,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        created_at=as_utc(row.created_at),
    )
