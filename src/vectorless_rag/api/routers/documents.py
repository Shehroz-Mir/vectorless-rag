"""Document routes (spec 7). Plain `def`: FastAPI runs them on its thread pool (spec 5.1)."""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Response, UploadFile, status

from vectorless_rag.api.dependencies import CurrentUser, Library
from vectorless_rag.models import DocumentOut, FigureOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def upload_document(file: UploadFile, response: Response, user_id: CurrentUser, library: Library) -> DocumentOut:
    """Upload a PDF: 202 with the queued document, or 200 with the existing one if this user uploaded
    the same file before."""
    data = file.file.read(library.max_upload_bytes + 1)  # one byte over the limit is enough to reject it
    upload = library.upload(user_id, file.filename or "", data)
    if not upload.created:
        response.status_code = status.HTTP_200_OK
    return DocumentOut.of(upload.document)


@router.get("")
def list_documents(user_id: CurrentUser, library: Library) -> list[DocumentOut]:
    """The user's documents, newest first."""
    return [DocumentOut.of(document) for document in library.list_documents(user_id)]


@router.get("/{document_id}")
def get_document(document_id: UUID, user_id: CurrentUser, library: Library) -> DocumentOut:
    return DocumentOut.of(library.get(user_id, document_id))


@router.get("/{document_id}/figures")
def list_figures(document_id: UUID, user_id: CurrentUser, library: Library) -> list[FigureOut]:
    """The stored figure descriptions, for review and debugging."""
    return [FigureOut.of(figure) for figure in library.figures_of(user_id, document_id)]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID, user_id: CurrentUser, library: Library) -> None:
    """Delete from PageIndex, the registry and disk."""
    library.delete(user_id, document_id)
