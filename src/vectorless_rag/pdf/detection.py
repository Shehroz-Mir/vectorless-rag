"""Figure-page detection (spec 5.3a; implements ports.DetectFigurePages once rules are bound).

Rule verified on the sample PDFs (Spike D): a placed image covering enough of the page is a raster
figure; drawing clusters with little text inside them are vector figures (ruled tables are dense
with text, so they are skipped). A drawing count alone cannot tell illustrations from tables.

The composition root binds the rules: `partial(detect_figure_pages, rules=DetectionRules(...))`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf

from vectorless_rag.models import BoundingBox, DetectedFigure, FigureKind, FigurePage
from vectorless_rag.pdf.mupdf_lock import with_mupdf_lock
from vectorless_rag.pdf.reading import page_text  # same package: full module path, never via pdf/__init__


@dataclass(frozen=True)
class DetectionRules:
    min_image_area_ratio: float  # a placed image at least this share of the page is a figure
    min_graphic_cluster_ratio: float  # smaller drawing clusters are icons, bullets or rules
    max_cluster_text_density: float  # text chars per 1% of page area; above this a cluster is a table
    min_vector_figure_area: float  # graphic clusters must add up to this share of the page


@with_mupdf_lock
def detect_figure_pages(pdf_path: Path, rules: DetectionRules) -> list[FigurePage]:
    with pymupdf.open(pdf_path) as doc:
        pages: list[FigurePage] = []
        for number, page in enumerate(doc.pages(), start=1):
            figures = _raster_figures(page, rules) + _vector_figures(page, rules)
            if figures:
                pages.append(FigurePage(page=number, figures=tuple(figures)))
        return pages


def _raster_figures(page: pymupdf.Page, rules: DetectionRules) -> list[DetectedFigure]:
    page_area = page.rect.get_area()
    figures: list[DetectedFigure] = []
    for image in page.get_images(full=True):
        for placed in page.get_image_rects(image[0]):
            visible = placed & page.rect
            if visible.get_area() / page_area >= rules.min_image_area_ratio:
                figures.append(DetectedFigure(kind=FigureKind.RASTER, box=_box(visible)))
    return figures


def _vector_figures(page: pymupdf.Page, rules: DetectionRules) -> list[DetectedFigure]:
    page_area = page.rect.get_area()
    graphic: list[pymupdf.Rect] = []
    for cluster in page.cluster_drawings():
        visible = cluster & page.rect
        share = visible.get_area() / page_area
        if share < rules.min_graphic_cluster_ratio:
            continue
        if len(page_text(page, visible)) / (share * 100) <= rules.max_cluster_text_density:
            graphic.append(visible)
    if sum(rect.get_area() for rect in graphic) / page_area < rules.min_vector_figure_area:
        return []
    return [DetectedFigure(kind=FigureKind.VECTOR, box=_box(rect)) for rect in graphic]


def _box(rect: pymupdf.Rect) -> BoundingBox:
    return (rect.x0, rect.y0, rect.x1, rect.y1)
