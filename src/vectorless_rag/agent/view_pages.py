"""The view_pages tool (spec 5.7): original page images go to the main agent, which reads them itself.

The images travel inside the tool result, which reaches the model only through the Responses API
(Spike B). Each image is an OpenAI image_url block: LangChain's standard image block drops `detail`.
"""
from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool

from vectorless_rag.models import PageImage
from vectorless_rag.operations import PageViewer, ViewPagesRejected

ImageDetail = Literal["low", "high", "auto"]


def view_pages_tool(view_pages: PageViewer, detail: ImageDetail) -> BaseTool:
    """The tool around a PageViewer that is already bound to the asking user."""

    def view_pages_images(doc_name: str, pages: str) -> str | list[dict[str, Any]]:
        """Show page images of a document so you can read figures, screenshots, drawings and exact values yourself.

        Args:
            doc_name: The document name, exactly as the other tools show it, e.g. "report.pdf".
            pages: Page numbers, e.g. "12", "12,13" or "12-13"; a few pages per call.
        """
        try:
            return page_image_blocks(view_pages(doc_name, pages), detail)
        except ViewPagesRejected as error:
            return str(error)  # the agent reads why and can ask again

    return StructuredTool.from_function(view_pages_images, name="view_pages", parse_docstring=True)


def page_image_blocks(images: Sequence[PageImage], detail: ImageDetail) -> list[dict[str, Any]]:
    """Per page, a text label and then the image."""
    blocks: list[dict[str, Any]] = []
    for image in images:
        url = "data:image/png;base64," + base64.b64encode(image.png).decode("ascii")
        blocks.append({"type": "text", "text": f"Document: {image.doc_name}, page {image.page}:"})
        blocks.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
    return blocks
