"""Spike C (Open Q5, Q6, light Q7): PageIndex agent tools in LangChain, and per-user isolation.

User A = Spike A's storage (TDI-110, already indexed). User B = a fresh storage_path with two
2-page PDFs cut from other manuals. Run Spike A first.
"""
from __future__ import annotations

import inspect
import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import pymupdf
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pageindex import PageIndexClient
from pageindex.errors import PageIndexAPIError

from common import CHAT_MODEL, INDEX_MODEL, NAVIO, TD_PILOT, out_path

USER_A_STORAGE = out_path("spike_a", "pageindex", "x").parent
USER_B_STORAGE = out_path("spike_c", "user_b", "x").parent
QUESTION = (
    "At what temperature does the TD I-110 shut itself off to avoid harm? "
    "Give the value in °C and °F and cite the page."
)


def cut_pdf(source: Path, first_page: int, last_page: int, target: Path) -> Path:
    with pymupdf.open(source) as src, pymupdf.open() as mini:
        mini.insert_pdf(src, from_page=first_page - 1, to_page=last_page - 1)
        mini.save(target)
    return target


def describe_tools(client: PageIndexClient) -> list[dict[str, Any]]:
    return [
        {
            "name": fn.__name__,
            "signature": str(inspect.signature(fn)),
            "doc_first_line": (fn.__doc__ or "").splitlines()[0],
            "doc_has_args_section": "Args:" in (fn.__doc__ or ""),
        }
        for fn in client.agent_tools()
    ]


def wrap_tools(client: PageIndexClient) -> list[StructuredTool]:
    return [StructuredTool.from_function(fn) for fn in client.agent_tools()]


def call(tools: list[StructuredTool], name: str, args: dict[str, Any]) -> dict[str, Any]:
    tool = next(t for t in tools if t.name == name)
    return json.loads(tool.invoke(args))


def expect_error(action: Any) -> str:
    try:
        action()
    except PageIndexAPIError as exc:
        return f"raised PageIndexAPIError: {exc}"
    return "NO ERROR (isolation gap!)"


def index_user_b_with_concurrent_reads() -> dict[str, Any]:
    """Index doc 1, then index doc 2 while another thread keeps reading the same storage_path."""
    shutil.rmtree(USER_B_STORAGE, ignore_errors=True)
    client_b = PageIndexClient(index={"model": INDEX_MODEL, "storage_path": str(USER_B_STORAGE), "summary_concurrency": 4})
    first = client_b.submit_document(str(cut_pdf(NAVIO, 13, 14, out_path("spike_c", "inputs", "navio-mini.pdf"))))

    reads = 0
    read_errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        nonlocal reads
        while not stop.is_set():
            try:
                listing = client_b.list_documents()
                for doc in listing["documents"]:
                    client_b.get_tree(doc["id"], node_summary=True, include_text=False)
                    client_b.get_page_content(doc["id"], "1")
                reads += 1
            except Exception as exc:  # spike: count any failure mode during the concurrent write
                read_errors.append(f"{type(exc).__name__}: {exc}")
            time.sleep(0.05)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    second = client_b.submit_document(str(cut_pdf(TD_PILOT, 24, 25, out_path("spike_c", "inputs", "pilot-mini.pdf"))))
    stop.set()
    thread.join()
    return {
        "user_b_docs": [first["name"], second["name"]],
        "reads_during_second_index": reads,
        "read_errors": read_errors[:5],
        "read_error_count": len(read_errors),
    }


def run_agent(client: PageIndexClient, tools: list[StructuredTool]) -> dict[str, Any]:
    system = client.agent_instructions() + "\n\n" + client.citation_prompt()
    model = ChatOpenAI(model=CHAT_MODEL, use_responses_api=True)
    agent = create_agent(model, tools=tools, system_prompt=system, middleware=[ModelCallLimitMiddleware(run_limit=10)])
    result = agent.invoke({"messages": [{"role": "user", "content": QUESTION}]})
    answer: str = result["messages"][-1].text
    calls = [c["name"] + " " + json.dumps(c["args"]) for m in result["messages"] for c in getattr(m, "tool_calls", [])]
    return {
        "tool_calls": calls,
        "answer": answer,
        "get_citations": client.get_citations(answer),
        "resolve_citations_answer": client.resolve_citations(answer)["answer"],
    }


def main() -> None:
    if not (USER_A_STORAGE / "docs").is_dir():
        raise SystemExit("Run spike_a_invisible_text.py first (user A's library comes from it).")
    client_a = PageIndexClient(index={"model": INDEX_MODEL, "storage_path": str(USER_A_STORAGE)})
    report: dict[str, Any] = {"agent_tools": describe_tools(client_a)}

    tools_a = wrap_tools(client_a)
    report["structured_tools"] = [
        {"name": t.name, "args_schema": t.args, "description_chars": len(t.description)} for t in tools_a
    ]

    report["concurrency"] = index_user_b_with_concurrent_reads()
    client_b = PageIndexClient(index={"model": INDEX_MODEL, "storage_path": str(USER_B_STORAGE)})
    tools_b = wrap_tools(client_b)

    a_doc = client_a.list_documents()["documents"][0]
    names_a = [d["name"] for d in call(tools_a, "browse_documents", {})["documents"]]
    names_b = [d["name"] for d in call(tools_b, "browse_documents", {})["documents"]]
    report["isolation"] = {
        "a_browse": names_a,
        "b_browse": names_b,
        "b_get_page_content_of_a_doc": call(tools_b, "get_page_content", {"doc_name": a_doc["name"], "pages": "1"}).get("error"),
        "b_get_document_of_a_doc": call(tools_b, "get_document", {"doc_name": a_doc["name"]}).get("error"),
        "b_document_context_with_a_doc_id": expect_error(lambda: client_b.document_context(a_doc["id"])),
        "b_sdk_get_page_content_with_a_doc_id": expect_error(lambda: client_b.get_page_content(a_doc["id"], "1")),
        "b_get_citations_for_a_doc": client_b.get_citations(f'x <cite doc="{a_doc["name"]}" page="1"/>'),
        "a_get_page_content_works": "success" in call(tools_a, "get_page_content", {"doc_name": a_doc["name"], "pages": "27"}),
    }
    report["agent_run_user_a"] = run_agent(client_a, tools_a)

    out_path("spike_c", "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
