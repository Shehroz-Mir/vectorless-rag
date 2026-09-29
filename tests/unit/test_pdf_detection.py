from pathlib import Path

from tests.sample_pdfs import DEFAULT_DETECTION_RULES, build_pdf, drawing, image, table, text
from vectorless_rag.models import FigureKind
from vectorless_rag.pdf import detect_figure_pages, read_page_texts


def test_reads_every_page_text_in_order(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "t.pdf", [[text("first page")], [], [text("third page")]])

    assert read_page_texts(pdf) == ["first page", "", "third page"]


def test_large_image_is_a_raster_figure_but_an_icon_is_not(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "r.pdf", [
        [text("screenshot below"), image((72, 100, 400, 400))],  # ~22% of the page
        [text("warning"), image((72, 100, 110, 138))],  # icon, ~0.3%
    ])

    pages = detect_figure_pages(pdf, DEFAULT_DETECTION_RULES)

    assert [(p.page, p.main_figure.kind) for p in pages] == [(1, FigureKind.RASTER)]
    x0, y0, x1, y1 = pages[0].main_figure.box
    assert (round(x0), round(y0), round(x1), round(y1)) == (72, 100, 400, 400)


def test_line_drawing_is_a_vector_figure_but_a_table_is_not(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "v.pdf", [
        [text("Illustration 1: the device"), drawing((100, 150, 450, 450))],
        [text("Technical specifications"), table((72, 150, 520, 400))],
        [text("plain text only")],
    ])

    pages = detect_figure_pages(pdf, DEFAULT_DETECTION_RULES)

    assert [(p.page, p.main_figure.kind) for p in pages] == [(1, FigureKind.VECTOR)]


def test_small_drawing_clusters_are_ignored(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "s.pdf", [[text("note"), drawing((72, 100, 100, 128))]])

    assert detect_figure_pages(pdf, DEFAULT_DETECTION_RULES) == []


def test_page_with_both_kinds_lists_every_figure(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "m.pdf", [[image((50, 50, 300, 250)), drawing((100, 400, 500, 800))]])

    [page] = detect_figure_pages(pdf, DEFAULT_DETECTION_RULES)

    assert {figure.kind for figure in page.figures} == {FigureKind.RASTER, FigureKind.VECTOR}
    assert page.main_figure.kind is FigureKind.VECTOR  # the larger one


def test_image_partly_off_the_page_counts_only_its_visible_part(tmp_path: Path) -> None:
    pdf = build_pdf(tmp_path / "o.pdf", [[image((580, 800, 900, 1200))]])  # only ~15×42 pt visible

    assert detect_figure_pages(pdf, DEFAULT_DETECTION_RULES) == []
