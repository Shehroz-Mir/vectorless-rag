"""Small generated PDFs for the pdf/ tests, and the detection rules they use. Pages are A4 (595 × 842 pt)."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pymupdf

from vectorless_rag.pdf import DetectionRules

DEFAULT_DETECTION_RULES = DetectionRules(
    min_image_area_ratio=0.03, min_graphic_cluster_ratio=0.01, max_cluster_text_density=5.0, min_vector_figure_area=0.02,
)
"""The spec 10 defaults, verified on the sample PDFs by Spike D."""

PageBuilder = Callable[[pymupdf.Page], None]


def build_pdf(path: Path, pages: Sequence[Sequence[PageBuilder]]) -> Path:
    """One page per entry; each entry lists what to draw on that page."""
    with pymupdf.open() as doc:
        for builders in pages:
            page = doc.new_page()
            for build in builders:
                build(page)
        doc.save(path)
    return path


def text(content: str, y: float = 72) -> PageBuilder:
    def build(page: pymupdf.Page) -> None:
        page.insert_text((72, y), content, fontsize=11)
    return build


def image(rect: tuple[float, float, float, float]) -> PageBuilder:
    """A placed raster image filling `rect`."""
    def build(page: pymupdf.Page) -> None:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 64, 64), False)
        pixmap.clear_with(120)
        page.insert_image(pymupdf.Rect(rect), pixmap=pixmap)
    return build


def drawing(rect: tuple[float, float, float, float]) -> PageBuilder:
    """A line drawing (curves and lines, no text) filling `rect`."""
    def build(page: pymupdf.Page) -> None:
        box = pymupdf.Rect(rect)
        shape = page.new_shape()
        shape.draw_rect(box)
        shape.draw_circle(box.tl + (box.width / 3, box.height / 2), min(box.width, box.height) / 5)
        shape.draw_bezier(box.bl, box.tl, box.br, box.tr)
        shape.draw_line(box.tl, box.br)
        shape.finish(color=(0, 0, 0), width=1)
        shape.commit()
    return build


def table(rect: tuple[float, float, float, float], rows: int = 4, cols: int = 3) -> PageBuilder:
    """A ruled table with text in every cell."""
    def build(page: pymupdf.Page) -> None:
        box = pymupdf.Rect(rect)
        shape = page.new_shape()
        for i in range(rows + 1):
            y = box.y0 + i * box.height / rows
            shape.draw_line((box.x0, y), (box.x1, y))
        for j in range(cols + 1):
            x = box.x0 + j * box.width / cols
            shape.draw_line((x, box.y0), (x, box.y1))
        shape.finish(color=(0, 0, 0), width=0.5)
        shape.commit()
        for i in range(rows):
            for j in range(cols):
                x = box.x0 + j * box.width / cols + 4
                y = box.y0 + i * box.height / rows + 14
                page.insert_text((x, y), f"Cell {i}-{j} value", fontsize=9)
    return build
