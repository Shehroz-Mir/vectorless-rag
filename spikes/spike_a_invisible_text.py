"""Spike A (Open Q2, Q3): does PageIndex local mode pick up invisible text (render_mode=3)?

Steps:
  1. Copy TDI-110 and write hand-written figure descriptions (with canary words) as
     invisible text on p13 (raster photo) and p14 (vector illustration).
  2. Free checks: page renders unchanged; PyMuPDF, PyPDF2 (get_page_content's
     extractor) and pdfium/Flash (tree extractor) all see the text; LLM-free Flash
     tree of original vs enriched is identical (Q3).
  3. One real local index: canary in get_page_content(), and node summaries.

Run with --no-index to skip step 3 (no API cost).
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf

from common import INDEX_MODEL, TDI_110, out_path

SPIKE_DIR = out_path("spike_a", "x").parent
ENRICHED_PDF = out_path("spike_a", "enriched", TDI_110.name)  # same name as the original (decision Q2)
STORAGE = SPIKE_DIR / "pageindex"


@dataclass(frozen=True)
class FigureNote:
    page: int  # 1-based
    kind: str  # "raster" | "vector"
    canary: str
    text: str


NOTES = [
    FigureNote(
        page=13, kind="raster", canary="OKAPI-4402",
        text=(
            "[FIGURE DESCRIPTION p13 fig1] Photo: front view of the TD I-110 tablet. "
            "A black tablet with a wide dark bezel, rounded corners and a large blank screen. "
            "The label 'TD I-110' is printed at the top right of the bezel and a row of small "
            "sensor dots sits at the top centre. Two small feet are visible on the bottom edge. "
            "Canary: OKAPI-4402."
        ),
    ),
    FigureNote(
        page=14, kind="vector", canary="ZEBRA-7731",
        text=(
            "[FIGURE DESCRIPTION p14 fig1] Line drawing: the TD I-110 tablet shown from the front "
            "and the sides, with numbered callouts 1 to 13 pointing at ports, sensors and buttons. "
            "The Power Button (1) and the Volume Buttons (8) are on the top edge; the switch ports "
            "(9, 10) and the Audio Jack Port (11) are on the side edge; the Windows Home Button (12) "
            "is on the front below the screen. Canary: ZEBRA-7731."
        ),
    ),
]


def figure_rect(page: pymupdf.Page, note: FigureNote) -> pymupdf.Rect:
    """Where the figure sits: the largest placed image, or the largest drawing cluster."""
    if note.kind == "raster":
        rects = [r for img in page.get_images(full=True) for r in page.get_image_rects(img[0])]
    else:
        rects = list(page.cluster_drawings())
    if not rects:
        raise RuntimeError(f"no {note.kind} figure found on page {note.page}")
    return max(rects, key=lambda r: r.width * r.height)


def write_invisible_notes(source: Path, target: Path, notes: list[FigureNote]) -> dict[int, list[float]]:
    placed: dict[int, list[float]] = {}
    with pymupdf.open(source) as doc:
        for note in notes:
            page = doc[note.page - 1]
            box = figure_rect(page, note)
            fontsize = 6.0
            while page.insert_textbox(box, note.text, fontsize=fontsize, fontname="helv", render_mode=3) < 0:
                fontsize -= 0.5  # insert_textbox writes nothing when the text does not fit
                if fontsize < 2:
                    raise RuntimeError(f"description does not fit on page {note.page}")
            placed[note.page] = [round(v, 1) for v in (*box, fontsize)]
        doc.save(target)
    return placed


def render_identical(original: Path, enriched: Path, page_no: int) -> bool:
    with pymupdf.open(original) as a, pymupdf.open(enriched) as b:
        pix_a = a[page_no - 1].get_pixmap(dpi=100)
        pix_b = b[page_no - 1].get_pixmap(dpi=100)
        return pix_a.samples == pix_b.samples


def flatten(nodes: list[dict[str, Any]], depth: int = 0) -> list[tuple[int, str, int, int]]:
    rows: list[tuple[int, str, int, int]] = []
    for node in nodes:
        rows.append((depth, str(node.get("title")), int(node.get("start_index") or 0), int(node.get("end_index") or 0)))
        rows.extend(flatten(node.get("nodes") or [], depth + 1))
    return rows


def free_checks() -> dict[str, Any]:
    from pageindex.flash import page_index_flash
    from pageindex.flash.api import _validate_pdf  # pyright: ignore[reportPrivateUsage]
    from pageindex.flash.main import extract_toc
    from pageindex.local_api import LocalAPI

    report: dict[str, Any] = {}
    pypdf2_pages = LocalAPI._extract_page_texts(str(ENRICHED_PDF))  # pyright: ignore[reportPrivateUsage]
    # page_index_flash() drops page_texts before returning; extract_toc() is what feeds the summaries.
    pdfium_pages: list[str] = extract_toc(_validate_pdf(str(ENRICHED_PDF)), use_embedded_toc=True).get("page_texts") or []
    flash_original = page_index_flash(str(TDI_110), summary=False, optimize=False)
    flash_enriched = page_index_flash(str(ENRICHED_PDF), summary=False, optimize=False)
    # PDFs without bookmarks rely on layout heading detection: compare that path too (Q3).
    detected_original = page_index_flash(str(TDI_110), summary=False, optimize=False, use_embedded_toc=False)
    detected_enriched = page_index_flash(str(ENRICHED_PDF), summary=False, optimize=False, use_embedded_toc=False)
    rows_detected_original = flatten(detected_original["structure"])
    rows_detected_enriched = flatten(detected_enriched["structure"])
    report["detected_tree (no bookmarks)"] = {
        "toc_source": [detected_original.get("toc_source"), detected_enriched.get("toc_source")],
        "identical": rows_detected_original == rows_detected_enriched,
        "nodes": [len(rows_detected_original), len(rows_detected_enriched)],
        "only_original": [r for r in rows_detected_original if r not in rows_detected_enriched],
        "only_enriched": [r for r in rows_detected_enriched if r not in rows_detected_original],
    }
    with pymupdf.open(ENRICHED_PDF) as doc:
        for note in NOTES:
            report[f"p{note.page}"] = {
                "render_identical": render_identical(TDI_110, ENRICHED_PDF, note.page),
                "pymupdf_sees_canary": note.canary in doc[note.page - 1].get_text(),
                "pypdf2_sees_canary (get_page_content path)": note.canary in pypdf2_pages[note.page - 1],
                "pdfium_sees_canary (tree/summary path)": (
                    note.canary in pdfium_pages[note.page - 1] if len(pdfium_pages) >= note.page else None
                ),
            }
    rows_original = flatten(flash_original["structure"])
    rows_enriched = flatten(flash_enriched["structure"])
    report["flash_toc_source"] = [flash_original.get("toc_source"), flash_enriched.get("toc_source")]
    report["flash_tree_identical"] = rows_original == rows_enriched
    report["flash_nodes"] = [len(rows_original), len(rows_enriched)]
    report["tree_diff"] = {
        "only_original": [r for r in rows_original if r not in rows_enriched],
        "only_enriched": [r for r in rows_enriched if r not in rows_original],
    }
    report["canary_in_any_title"] = any(n.canary in r[1] or "FIGURE DESCRIPTION" in r[1] for n in NOTES for r in rows_enriched)
    report["pypdf2_p14_excerpt"] = pypdf2_pages[13][-420:]
    return report


def indexed_checks() -> dict[str, Any]:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger
    from pageindex import PageIndexClient

    class UsageCounter(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0

        def _add(self, response_obj: Any) -> None:
            usage = getattr(response_obj, "usage", None)
            self.calls += 1
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

        def log_success_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            self._add(response_obj)

        async def async_log_success_event(self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any) -> None:
            self._add(response_obj)

    counter = UsageCounter()
    litellm.callbacks = [counter]
    shutil.rmtree(STORAGE, ignore_errors=True)
    client = PageIndexClient(index={"model": INDEX_MODEL, "storage_path": str(STORAGE)})

    started = time.perf_counter()
    submitted = client.submit_document(str(ENRICHED_PDF))
    elapsed = time.perf_counter() - started
    doc_id = submitted["doc_id"]

    pages = client.get_page_content(doc_id, "13-14")
    page_text = {p["page_index"]: p["markdown"] for p in pages}
    tree = client.get_tree(doc_id, node_summary=True, include_text=False)["result"]

    covering: list[dict[str, Any]] = []

    def walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            summary = node.get("summary") or node.get("prefix_summary") or ""
            if any(n.canary in summary for n in NOTES) or node.get("page_index") in (12, 13, 14):
                covering.append({"title": node.get("title"), "page_index": node.get("page_index"), "summary": summary})
            walk(node.get("nodes") or [])

    walk(tree)
    all_summaries = json.dumps(tree, ensure_ascii=False)
    return {
        "stored_name": submitted["name"],
        "doc_id": doc_id,
        "index_seconds": round(elapsed, 1),
        "llm_calls": counter.calls,
        "prompt_tokens": counter.prompt_tokens,
        "completion_tokens": counter.completion_tokens,
        "get_page_content_has_canary": {f"p{n.page}": n.canary in page_text.get(n.page, "") for n in NOTES},
        "canary_in_tree_summaries": {n.canary: n.canary in all_summaries for n in NOTES},
        "figure_words_in_summaries": [w for w in ("photo", "line drawing", "callout", "figure") if w in all_summaries.lower()],
        "nodes_near_p12_14": covering,
        "top_level_titles": [n.get("title") for n in tree],
    }


def main() -> None:
    placed = write_invisible_notes(TDI_110, ENRICHED_PDF, NOTES)
    report: dict[str, Any] = {"placed_box_and_fontsize": placed, "free_checks": free_checks()}
    if "--no-index" not in sys.argv:
        report["indexed_checks"] = indexed_checks()
    out_path("spike_a", "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
