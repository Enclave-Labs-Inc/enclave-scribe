"""The agent — plans and executes OCR extraction per document.

Simplified 2026-08-27 flow:
    PDF → page images → for each page, one VLM call → assemble markdown → validate

No layout detection. No per-region routing. The VLM does everything from
a single page-level prompt. See scribe/agent/prompts.py for the rationale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import tools
from .models import ModelRegistry


@dataclass
class PageResult:
    page_index: int
    markdown: str
    elapsed_s: float


@dataclass
class DocumentResult:
    pdf_path: str
    n_pages: int
    pages: list[PageResult] = field(default_factory=list)
    markdown: str = ""
    validation: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


class Agent:
    """Orchestrates the OCR pipeline for a document."""

    def __init__(self, registry: ModelRegistry | None = None, verbose: bool = True):
        self.registry = registry or ModelRegistry()
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[agent] {msg}", flush=True)

    def parse_page(self, image, page_index: int) -> PageResult:
        t0 = time.time()
        markdown = tools.extract_page(image, self.registry)
        return PageResult(
            page_index=page_index,
            markdown=markdown,
            elapsed_s=time.time() - t0,
        )

    def parse(self, pdf_path: str | Path, dpi: int = 200) -> DocumentResult:
        t0 = time.time()
        pdf_path = str(pdf_path)
        self._log(f"loading PDF: {pdf_path}")
        pages = tools.pdf_to_pages(pdf_path, dpi=dpi)
        self._log(f"loaded {len(pages)} page(s) at {dpi} DPI")

        # Trigger VLM load once so per-page timings don't include it.
        self._log("loading VLM (first call may take a while)...")
        _ = self.registry.vlm()
        self._log(f"VLM ready on device={self.registry.vlm().device}")

        result = DocumentResult(pdf_path=pdf_path, n_pages=len(pages))
        for i, page_img in enumerate(pages, start=1):
            page_result = self.parse_page(page_img, page_index=i)
            result.pages.append(page_result)
            self._log(
                f"page {i}: {len(page_result.markdown):,} chars "
                f"in {page_result.elapsed_s:.1f}s"
            )

        result.markdown = tools.assemble_markdown([p.markdown for p in result.pages])
        result.validation = tools.validate_output(result.markdown, pages)
        result.elapsed_s = time.time() - t0

        self._log(
            f"done: {result.n_pages} pages, {result.validation['n_chars']:,} "
            f"chars, {len(result.validation['issues'])} issues, "
            f"{result.elapsed_s:.1f}s total"
        )
        if result.validation["issues"]:
            self._log(f"issues: {result.validation['issues']}")
        return result
