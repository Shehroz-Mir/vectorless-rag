"""Run the service: `python -m vectorless_rag.api` (http://127.0.0.1:8000, docs at /docs)."""
import uvicorn

# PageIndex starts worker processes while indexing; on Windows each one re-imports __main__.
if __name__ == "__main__":
    uvicorn.run("vectorless_rag.api.main:app", host="127.0.0.1", port=8000)
