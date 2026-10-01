"""The question route (spec 7). Plain `def`: the agent is synchronous and runs on FastAPI's thread pool."""
from __future__ import annotations

from fastapi import APIRouter

from vectorless_rag.api.dependencies import CurrentUser, Questions
from vectorless_rag.models import FullQueryResponse, QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


# The full response comes first: a brief one would also accept a full one and drop its steps.
@router.post("/query")
def ask(query: QueryRequest, user_id: CurrentUser, questions: Questions) -> FullQueryResponse | QueryResponse:
    """Answer a question over the selected documents, or over all of the user's documents.
    `"detail": "full"` adds the steps the agent took (agent-runs spec 2.5)."""
    return questions.answer(user_id, query)
