"""Run the eval (agent-runs spec 4.2): `python -m evals.run [--only ID ...] [--repeat N]`.

Builds the eval library once under evals/var/ with the service's own upload and ingestion (figure
descriptions included), asks every question through QuestionAnswering with detail="full" and without
document_ids, scores and judges each answer, and writes results.jsonl and summary.md to
evals/results/<date>/. The eval's composition root, like api/ and worker/ for the service.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from evals.config import load_eval_settings
from evals.judge import Evidence, Judge, OpenAiJudge, PageImageEvidence, PageText
from evals.metrics import QuestionResult, generation_scores, navigation_cost, retrieval_scores, tool_steps
from evals.questions import EvalQuestion, PageRef, load_questions, select
from evals.report import RunInfo, passed, summary
from vectorless_rag.api import Services, build_services
from vectorless_rag.config import Settings, load_settings
from vectorless_rag.models import (
    VIEW_PAGES_TOOL,
    AgentRun,
    Document,
    DocumentStatus,
    FullQueryResponse,
    QueryRequest,
    ToolOutcome,
)
from vectorless_rag.operations import AnswerIncomplete, PageRenderer, QuestionAnswering, ReadPageTexts
from vectorless_rag.pdf import PyMuPdfPageRenderer, read_page_texts

logger = logging.getLogger("evals")

EVAL_ROOT = Path("evals/var")  # the eval library: uploads, enriched copies, PageIndex storage, SQLite
RESULTS_ROOT = Path("evals/results")
SAMPLES = Path("Data")  # the sample manuals; only read
EVAL_USER = "eval"
PAGE_TEXT_TOOL = "get_page_content"
BUILD_POLL_S = 5.0
BUILD_TIMEOUT_S = 3_600.0
EMPTY_RUN = AgentRun.of("", [], duration_ms=0)


@dataclass(frozen=True)
class Attempt:
    answer: str
    citations: tuple[PageRef, ...]
    run: AgentRun
    error: str | None = None


def eval_settings(settings: Settings, root: Path = EVAL_ROOT) -> Settings:
    """The service's settings with the eval library's own data root and database."""
    return settings.model_copy(update={"data_root": root, "database_url": f"sqlite:///{(root / 'eval.db').resolve().as_posix()}"})


def build_library(services: Services, samples: Path = SAMPLES) -> dict[str, Document]:
    """Upload every sample manual for the eval user (an unchanged file is not uploaded again) and run the
    ingestion worker until each is indexed. Returns the eval user's documents by file name."""
    for pdf in sorted(samples.glob("*.pdf")):
        services.library.upload(EVAL_USER, pdf.name, pdf.read_bytes())
    if _unfinished(services.library.list_documents(EVAL_USER)):
        if services.worker is None:
            raise RuntimeError("the services have no ingestion worker")
        services.worker.start()
        try:
            _wait_for_ingestion(services)
        finally:
            services.worker.stop()
    documents = services.library.list_documents(EVAL_USER)
    failed = [d for d in documents if d.status is DocumentStatus.FAILED]
    if failed:
        names = "; ".join(f"{d.filename}: {d.error}" for d in failed)
        raise RuntimeError(f"could not index {names}. Delete {EVAL_ROOT} to rebuild the library.")
    return {d.filename: d for d in documents}


def _unfinished(documents: Iterable[Document]) -> list[Document]:
    return [d for d in documents if d.status not in (DocumentStatus.COMPLETED, DocumentStatus.FAILED)]


def _wait_for_ingestion(services: Services) -> None:
    deadline, last = time.monotonic() + BUILD_TIMEOUT_S, ""
    while True:
        documents = services.library.list_documents(EVAL_USER)
        statuses = ", ".join(f"{d.filename}: {d.status}" for d in documents)
        if statuses != last:
            logger.info("eval library: %s", statuses)
            last = statuses
        if not _unfinished(documents):
            return
        if time.monotonic() > deadline:
            raise TimeoutError(f"the eval library was not built within {BUILD_TIMEOUT_S:g} s")
        time.sleep(BUILD_POLL_S)


class EvidenceReader:
    """What the agent saw in a run: the text of the pages it read (from the enriched copy, which is what
    PageIndex indexed, figure descriptions included) and images of the pages it viewed (the original)."""

    def __init__(self, library: Mapping[str, Document], renderer: PageRenderer, read_texts: ReadPageTexts = read_page_texts) -> None:
        self._library = library
        self._renderer = renderer
        self._read_texts = read_texts
        self._texts: dict[str, list[str]] = {}

    def evidence(self, run: AgentRun) -> Evidence:
        read: dict[PageRef, None] = {}
        viewed: dict[PageRef, None] = {}
        for step in tool_steps(run):
            document = step.document
            if step.outcome is not ToolOutcome.OK or document is None or document not in self._library:
                continue
            target = viewed if step.tool == VIEW_PAGES_TOOL else read if step.tool == PAGE_TEXT_TOOL else {}
            for page in step.pages:
                target.setdefault((document, page))
        return Evidence(
            texts=[PageText(document, page, self._page_text(document, page)) for document, page in read],
            images=self._images(list(viewed)),
        )

    def _page_text(self, name: str, page: int) -> str:
        if name not in self._texts:
            document = self._library[name]
            self._texts[name] = self._read_texts(document.enriched_path or document.original_path)
        texts = self._texts[name]
        return texts[page - 1] if 1 <= page <= len(texts) else ""

    def _images(self, pages: Sequence[PageRef]) -> list[PageImageEvidence]:
        images: list[PageImageEvidence] = []
        for name in dict.fromkeys(document for document, _ in pages):
            numbers = [page for document, page in pages if document == name]
            pngs = self._renderer.render_png(self._library[name].original_path, numbers)
            images += [PageImageEvidence(name, page, png) for page, png in zip(numbers, pngs, strict=True)]
        return images


def ask(questions: QuestionAnswering, question: EvalQuestion) -> Attempt:
    """The question through the service's own use case, as a user would ask it: no document_ids."""
    try:
        response = questions.answer(EVAL_USER, QueryRequest(question=question.question, detail="full"))
    except AnswerIncomplete as error:
        return Attempt("", (), error.run or EMPTY_RUN, f"incomplete: {error}")
    except Exception as error:  # one failing question must not end a long run
        logger.exception("question %s failed", question.id)
        return Attempt("", (), EMPTY_RUN, f"{type(error).__name__}: {error}")
    if not isinstance(response, FullQueryResponse):
        raise TypeError("a full query must return a FullQueryResponse")
    run = AgentRun(answer=response.answer, steps=response.steps, totals=response.stats)
    return Attempt(response.answer, tuple((c.filename, c.page) for c in response.citations), run)


def evaluate(
    question: EvalQuestion, repeat: int, *, questions: QuestionAnswering, judge: Judge, evidence: EvidenceReader, chat_model: str,
) -> QuestionResult:
    attempt = ask(questions, question)
    verdict, judge_error, judge_cost = None, None, None
    if attempt.answer.strip():
        try:
            judgement = judge.judge(question, attempt.answer, attempt.citations, evidence.evidence(attempt.run))
            verdict, judge_cost = judgement.verdict, judgement.cost_usd
        except Exception as error:  # the answer and its retrieval scores still count
            logger.exception("judging %s failed", question.id)
            judge_error = f"{type(error).__name__}: {error}"
    return QuestionResult(
        id=question.id, type=question.type, repeat=repeat, question=question.question, reference=question.answer,
        answer=attempt.answer, citations=attempt.citations, error=attempt.error, run=attempt.run,
        retrieval=retrieval_scores(question, attempt.run),
        navigation=navigation_cost(attempt.run, chat_model),
        generation=generation_scores(question, attempt.answer, attempt.citations, verdict),
        verdict=verdict, judge_error=judge_error, judge_cost_usd=judge_cost,
    )


def results_folder(root: Path, day: date) -> Path:
    """<root>/<date>/, or <date>-2, -3 … when that day already has a run: nothing is overwritten."""
    folder, number = root / day.isoformat(), 1
    while folder.exists():
        number += 1
        folder = root / f"{day.isoformat()}-{number}"
    return folder


def write_results(results: Sequence[QuestionResult], info: RunInfo, root: Path = RESULTS_ROOT) -> Path:
    folder = results_folder(root, info.started_at.date())
    folder.mkdir(parents=True)
    (folder / "results.jsonl").write_bytes("".join(r.model_dump_json() + "\n" for r in results).encode("utf-8"))
    (folder / "summary.md").write_bytes(summary(results, info).encode("utf-8"))
    return folder


def run_settings(settings: Settings) -> dict[str, str]:
    """The settings that shape a run, for the summary."""
    names = (
        "agent_max_steps", "agent_timeout_s", "agent_reasoning_summary", "view_pages_max_calls",
        "view_pages_max_pages", "max_image_sets_in_context", "view_pages_image_detail", "render_dpi",
    )
    return {name.upper(): str(getattr(settings, name)) for name in names}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m evals.run", description="Run the eval set against the real service.")
    parser.add_argument("--only", nargs="+", default=[], metavar="ID", help="ask only these questions")
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="ask each question N times (default 1)")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    try:
        chosen = select(load_questions(), args.only)
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("httpx", "LiteLLM", "litellm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    started_at, started = datetime.now(), time.monotonic()
    settings = eval_settings(load_settings())
    judge_model = load_eval_settings().eval_judge_model
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    services = build_services(settings)
    library = build_library(services)
    judge = OpenAiJudge(settings.openai_api_key.get_secret_value(), judge_model)
    evidence = EvidenceReader(library, PyMuPdfPageRenderer(settings.render_dpi))

    planned = [(question, repeat) for question in chosen for repeat in range(args.repeat)]
    results: list[QuestionResult] = []
    for number, (question, repeat) in enumerate(planned, start=1):
        result = evaluate(question, repeat, questions=services.questions, judge=judge, evidence=evidence, chat_model=settings.chat_model)
        logger.info("[%d/%d] %s: %s", number, len(planned), question.id, "pass" if passed(result) else "fail")
        results.append(result)

    info = RunInfo(
        started_at=started_at, duration_s=time.monotonic() - started, repeat=args.repeat,
        chat_model=settings.chat_model, judge_model=judge_model, index_model=settings.index_model,
        vision_model=settings.vision_model, settings=run_settings(settings),
    )
    folder = write_results(results, info)
    print(f"Wrote {folder / 'summary.md'} and {folder / 'results.jsonl'}")


if __name__ == "__main__":  # PageIndex starts worker processes, which import this module again on Windows
    main()
