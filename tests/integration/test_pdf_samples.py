"""Detection on the real sample PDFs in Data/ must reproduce Spike D (docs/spike-findings.md).

Data/ is read-only input: these tests only open the files.
"""
from pathlib import Path

import pytest

from vectorless_rag.models.figures import FigureKind
from vectorless_rag.pdf.detection import DetectionRules, detect_figure_pages

SAMPLES = Path(__file__).resolve().parents[2] / "Data"
RULES = DetectionRules(
    min_image_area_ratio=0.03, min_graphic_cluster_ratio=0.01, max_cluster_text_density=5.0, min_vector_figure_area=0.02,
)

# Page lists from Spike D; the vector lists match the 28 illustrations labelled by eye.
EXPECTED = {
    "TD_I-Series I-13_I-16_UsersManual_en-US_1000280.pdf": (
        [1, 14, 27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 40, 41, 42, 43, 44, 45, 46, 47],
        [11, 17, 18, 19, 20, 21, 22, 23, 26, 38, 62],
    ),
    "TD_Navio_UsersManual_en-US_1000965-01.pdf": ([1], [6, 12, 13, 14, 15]),
    "TDI-110_UsersManual_en-US_WEB_1000958-01.pdf": ([1, 13, 19], [7, 14, 15, 27]),
    "TobiiDynavox_TDPilot_UsersManual_en-GB_1001335-20.pdf": (
        [1, 20, 21, 22, 25, 28, 29, 30, 32, 33, 34],
        [8, 14, 15, 23, 24, 25, 26],
    ),
}

pytestmark = pytest.mark.skipif(not SAMPLES.is_dir(), reason="sample PDFs in Data/ not present")


@pytest.mark.parametrize("filename", sorted(EXPECTED))
def test_detection_matches_spike_d(filename: str) -> None:
    raster_expected, vector_expected = EXPECTED[filename]

    pages = detect_figure_pages(SAMPLES / filename, RULES)

    raster = [p.page for p in pages if any(f.kind is FigureKind.RASTER for f in p.figures)]
    vector = [p.page for p in pages if any(f.kind is FigureKind.VECTOR for f in p.figures)]
    assert (raster, vector) == (raster_expected, vector_expected)


def test_sixty_of_180_pages_are_figure_pages() -> None:
    total = sum(len(detect_figure_pages(SAMPLES / name, RULES)) for name in EXPECTED)

    assert total == 60
