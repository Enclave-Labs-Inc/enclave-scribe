"""The agent — plans and executes OCR extraction per document.

Flow per document:
  1. PDF → page images
  2. For each page:
       a. Detect layout regions
       b. Order regions by reading order
       c. For each region: crop, route to the right extractor by label
       d. Collect (label, extracted_content) pairs
  3. Assemble all pages into one markdown document
  4. Validate the assembled output; log any issues

The orchestrator DOES NOT know how any tool is implemented — it only calls
functions from tools.py. That separation lets us swap the layout detector,
the VLM, or the assembly logic independently.
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
    n_regions: int
    extractions: list[tuple[str, str]]   # [(label, content), ...] in reading order
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
        """Detect regions on a page and extract each one."""
        t0 = time.time()
        regions = tools.detect_layout(image, self.registry)
        regions = tools.order_regions(regions)
        self._log(f"page {page_index}: {len(regions)} region(s) detected")

        extractions: list[tuple[str, str]] = []
        for i, r in enumerate(regions):
            label = r["label"]
            self._log(f"  region {i} [{label}] bbox={r['bbox']}")
            crop = tools.crop_region(image, r["bbox"])
            content = tools.extract_for_label(label, crop, self.registry)
            extractions.append((label, content))

        return PageResult(
            page_index=page_index,
            n_regions=len(regions),
            extractions=extractions,
            elapsed_s=time.time() - t0,
        )

    def parse(self, pdf_path: str | Path, dpi: int = 200) -> DocumentResult:
        """Run the full pipeline on a PDF and return a DocumentResult."""
        t0 = time.time()
        pdf_path = str(pdf_path)
        self._log(f"loading PDF: {pdf_path}")
        pages = tools.pdf_to_pages(pdf_path, dpi=dpi)
        self._log(f"loaded {len(pages)} page(s) at {dpi} DPI")

        # Trigger VLM load once now so per-page timings don't include it.
        self._log("loading VLM (first call may take a while)...")
        _ = self.registry.vlm()
        self._log(f"VLM ready on device={self.registry.vlm().device}")

        result = DocumentResult(pdf_path=pdf_path, n_pages=len(pages))
        for i, page_img in enumerate(pages, start=1):
            page_result = self.parse_page(page_img, page_index=i)
            result.pages.append(page_result)
            self._log(
                f"page {i} done in {page_result.elapsed_s:.1f}s "
                f"({page_result.n_regions} regions)"
            )

        # Assemble
        per_page_extractions = [p.extractions for p in result.pages]
        result.markdown = tools.assemble_markdown(per_page_extractions)
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
