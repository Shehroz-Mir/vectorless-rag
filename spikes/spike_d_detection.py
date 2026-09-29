"""Spike D (Open Q4): figure-page detection on the real PDFs (no API cost).

Rule found here (see docs/spike-findings.md):
  raster figure page: a placed image covers >= RASTER_MIN_RATIO of the page
  vector figure page: drawing clusters that cover >= CLUSTER_MIN_RATIO of the page each and hold
                      <= CLUSTER_MAX_TEXT_DENSITY text chars per 1% of page area ("graphics, not a
                      ruled table") add up to >= VECTOR_MIN_AREA of the page
A plain drawing count cannot separate illustrations from ruled tables or pages with warning icons.

Run with --sheets to also write contact sheets of candidate pages for labelling by eye.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass

import pymupdf
from PIL import Image, ImageDraw

from common import I_SERIES, NAVIO, TD_PILOT, TDI_110, out_path

PDFS = {"iseries": I_SERIES, "navio": NAVIO, "tdi110": TDI_110, "pilot": TD_PILOT}

RASTER_MIN_RATIO = 0.03
CLUSTER_MIN_RATIO = 0.01
CLUSTER_MAX_TEXT_DENSITY = 5.0
VECTOR_MIN_AREA = 0.02

# Vector illustrations labelled by eye from the contact sheets (pilot p23 added after review:
# it has three small device drawings).
ILLUSTRATION_LABELS = {
    "iseries": {11, 17, 18, 19, 20, 21, 22, 23, 26, 27, 38, 62},
    "navio": {6, 12, 13, 14, 15},
    "tdi110": {7, 14, 15, 27},
    "pilot": {8, 14, 15, 23, 24, 25, 26},
}

THUMB_DPI = 40
SHEET_COLUMNS = 6
SHEET_SIZE = 18


@dataclass
class PageMetrics:
    doc: str
    page: int
    text_chars: int
    raster_max_ratio: float
    drawings: int
    curve_items: int
    graphic_cluster_area: float  # summed area share of low-text drawing clusters

    @property
    def raster_figure(self) -> bool:
        return self.raster_max_ratio >= RASTER_MIN_RATIO

    @property
    def vector_figure(self) -> bool:
        return self.graphic_cluster_area >= VECTOR_MIN_AREA


def page_text(page: pymupdf.Page, clip: pymupdf.Rect | None = None) -> str:
    """get_text() is typed as a union of all output formats; plain text is always str."""
    return str(page.get_text("text", clip=clip)).strip()


def page_metrics(doc_key: str, page: pymupdf.Page, page_no: int) -> PageMetrics:
    page_area = page.rect.width * page.rect.height
    raster_ratios = [
        (r & page.rect).get_area() / page_area
        for img in page.get_images(full=True)
        for r in page.get_image_rects(img[0])
    ]
    drawings = page.get_drawings()
    graphic_area = 0.0
    for cluster in page.cluster_drawings(drawings=drawings):
        cluster &= page.rect
        ratio = cluster.get_area() / page_area
        if ratio < CLUSTER_MIN_RATIO:
            continue  # icons, bullets, single rules
        if len(page_text(page, cluster)) / (ratio * 100) <= CLUSTER_MAX_TEXT_DENSITY:
            graphic_area += ratio
    return PageMetrics(
        doc=doc_key,
        page=page_no,
        text_chars=len(page_text(page)),
        raster_max_ratio=round(max(raster_ratios, default=0.0), 4),
        drawings=len(drawings),
        curve_items=sum(1 for d in drawings for item in d["items"] if item[0] in ("c", "qu")),
        graphic_cluster_area=round(graphic_area, 4),
    )


def contact_sheets(pages: list[PageMetrics]) -> list[str]:
    paths: list[str] = []
    docs = {key: pymupdf.open(path) for key, path in PDFS.items()}
    try:
        for start in range(0, len(pages), SHEET_SIZE):
            thumbs: list[Image.Image] = []
            for m in pages[start:start + SHEET_SIZE]:
                pix = docs[m.doc][m.page - 1].get_pixmap(dpi=THUMB_DPI)
                thumb = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                ImageDraw.Draw(thumb).rectangle((0, 0, thumb.width, 14), fill="yellow")
                ImageDraw.Draw(thumb).text((3, 1), f"{m.doc} p{m.page}", fill="black")
                thumbs.append(thumb)
            width, height = max(t.width for t in thumbs), max(t.height for t in thumbs)
            rows = (len(thumbs) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
            sheet = Image.new("RGB", (width * SHEET_COLUMNS, height * rows), "white")
            for i, thumb in enumerate(thumbs):
                sheet.paste(thumb, ((i % SHEET_COLUMNS) * width, (i // SHEET_COLUMNS) * height))
            path = out_path("spike_d", f"sheet_{start // SHEET_SIZE + 1:02d}.png")
            sheet.save(path)
            paths.append(str(path))
    finally:
        for doc in docs.values():
            doc.close()
    return paths


def main() -> None:
    metrics: list[PageMetrics] = []
    for key, path in PDFS.items():
        with pymupdf.open(path) as doc:
            metrics.extend(page_metrics(key, page, i + 1) for i, page in enumerate(doc.pages()))
    out_path("spike_d", "metrics.json").write_text(json.dumps([asdict(m) for m in metrics], indent=1), encoding="utf-8")

    labelled = [m for m in metrics if m.page in ILLUSTRATION_LABELS[m.doc]]
    missed = [(m.doc, m.page) for m in labelled if not (m.vector_figure or m.raster_figure)]
    extra = [(m.doc, m.page) for m in metrics if m.vector_figure and m.page not in ILLUSTRATION_LABELS[m.doc]]
    print(f"vector illustrations labelled: {len(labelled)}; missed by the rule: {missed}; "
          f"vector hits not labelled: {extra}")
    print("doc      pages  raster-fig  vector-fig  either  (page lists)")
    for key in PDFS:
        doc_pages = [m for m in metrics if m.doc == key]
        raster = [m.page for m in doc_pages if m.raster_figure]
        vector = [m.page for m in doc_pages if m.vector_figure]
        either = sorted(set(raster) | set(vector))
        print(f"{key:<8} {len(doc_pages):>5}  {len(raster):>10}  {len(vector):>10}  {len(either):>6}  "
              f"raster={raster} vector={vector}")
    total = sum(1 for m in metrics if m.raster_figure or m.vector_figure)
    print(f"figure pages: {total} of {len(metrics)}")
    if "--sheets" in sys.argv:
        candidates = [m for m in metrics if m.raster_max_ratio >= 0.01 or m.drawings >= 30]
        for sheet in contact_sheets(candidates):
            print("sheet:", sheet)


if __name__ == "__main__":
    main()
