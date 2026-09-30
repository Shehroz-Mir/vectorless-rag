"""The question route (spec 7). Plain `def`: the agent is synchronous and runs on FastAPI's thread pool."""
from __future__ import annotations

from fastapi import APIRouter

from vectorless_rag.api.dependencies import CurrentUser, Questions
from vectorless_rag.models import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query")
def ask(query: QueryRequest, user_id: CurrentUser, questions: Questions) -> QueryResponse:
    """Answer a question over the selected documents, or over all of the user's documents."""
    return questions.answer(user_id, query)
