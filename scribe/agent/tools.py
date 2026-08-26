"""Tool implementations for the agentic OCR pipeline.

Each tool does ONE thing and is independently callable by the orchestrator.
Tools accept PIL Images (or paths) and return either strings or structured
dicts — no hidden state, no orchestration logic here.

Design:
- All VLM-based tools go through _vlm_generate() so the model call is one code
  path (easier to tune, log, cache later).
- Cropping a region from a page image is a tool too (crop_region), so the
  orchestrator can compose "detect layout → crop each region → route to
  extractor" without any tool needing to know about the others.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from . import prompts
from .models import ModelRegistry, VLMHandle


# ─── PDF handling ───────────────────────────────────────────────────────────

def pdf_to_pages(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF into a list of PIL Images, one per page.

    Uses PyMuPDF (fitz) — no external system dependencies (unlike pdf2image
    which needs poppler). DPI 200 is a decent tradeoff: readable small text,
    keeps images under ~5 MB per page.
    """
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    pages: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    doc.close()
    return pages


# ─── Layout detection ───────────────────────────────────────────────────────

def detect_layout(image: Image.Image, registry: ModelRegistry) -> list[dict]:
    """Detect regions in a page image.

    Returns a list of dicts: [{label, bbox: (x1,y1,x2,y2), confidence}, ...]
    Labels are one of: text, table, figure, seal, logo, chart, image, picture.
    """
    detector = registry.layout()
    return detector.detect(image)


def crop_region(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    """Crop a bounding box out of a page image. Bbox is (x1, y1, x2, y2)."""
    return image.crop(bbox)


# ─── Region extractors (all go through the VLM) ─────────────────────────────

def _vlm_generate(
    handle: VLMHandle,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 2048,
) -> str:
    """Single code path for every VLM call.

    Kept generic so we can swap prompts/images without any tool knowing
    about generation params or chat templates.
    """
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    text = handle.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = handle.processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    ).to(handle.model.device)

    with torch.no_grad():
        output_ids = handle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return handle.processor.batch_decode(
        output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip()


def extract_text(image: Image.Image, registry: ModelRegistry) -> str:
    """Extract all text from a region, preserving original script and layout."""
    return _vlm_generate(registry.vlm(), image, prompts.TEXT)


def extract_table(image: Image.Image, registry: ModelRegistry) -> str:
    """Extract a table region as HTML with rowspan/colspan preserved."""
    return _vlm_generate(registry.vlm(), image, prompts.TABLE, max_new_tokens=4096)


def describe_image(image: Image.Image, registry: ModelRegistry) -> str:
    """Describe a figure, chart, or photograph in one to three sentences."""
    return _vlm_generate(registry.vlm(), image, prompts.FIGURE, max_new_tokens=512)


def identify_seal(image: Image.Image, registry: ModelRegistry) -> str:
    """Identify an official seal, emblem, or logo."""
    return _vlm_generate(registry.vlm(), image, prompts.SEAL, max_new_tokens=256)


def extract_chart_data(image: Image.Image, registry: ModelRegistry) -> str:
    """Extract data from a chart as a markdown table with chart-type header."""
    return _vlm_generate(registry.vlm(), image, prompts.CHART_DATA, max_new_tokens=1024)


# Router — orchestrator calls this to send a region to the right extractor.
_EXTRACTORS = {
    "text":    extract_text,
    "table":   extract_table,
    "figure":  describe_image,
    "image":   describe_image,
    "picture": describe_image,
    "seal":    identify_seal,
    "logo":    identify_seal,
    "chart":   extract_chart_data,
}


def extract_for_label(label: str, image: Image.Image, registry: ModelRegistry) -> str:
    """Route a region to the appropriate extractor based on its label."""
    fn = _EXTRACTORS.get(label.lower(), extract_text)
    return fn(image, registry)


# ─── Reading order ──────────────────────────────────────────────────────────

def order_regions(regions: list[dict]) -> list[dict]:
    """Sort regions in reading order (top-to-bottom, left-to-right within rows).

    Simple algorithm: sort by y-center, and within a horizontal band (regions
    whose y-centers are close), sort by x-center. Good enough for single- and
    two-column layouts. For complex multi-column layouts we'd need XY-cut.
    """
    def y_center(r): return (r["bbox"][1] + r["bbox"][3]) / 2
    def x_center(r): return (r["bbox"][0] + r["bbox"][2]) / 2
    def height(r):   return r["bbox"][3] - r["bbox"][1]

    if not regions:
        return regions

    # Sort by y-center first
    sorted_regions = sorted(regions, key=y_center)

    # Group into horizontal bands: two regions are in the same band if their
    # y-centers are within half the shorter one's height.
    bands: list[list[dict]] = [[sorted_regions[0]]]
    for r in sorted_regions[1:]:
        last_band_ymax = max(x["bbox"][3] for x in bands[-1])
        if r["bbox"][1] < last_band_ymax:
            bands[-1].append(r)
        else:
            bands.append([r])

    # Within each band, sort left-to-right
    out: list[dict] = []
    for band in bands:
        out.extend(sorted(band, key=x_center))
    return out


# ─── Markdown assembly ──────────────────────────────────────────────────────

def assemble_markdown(
    page_extractions: list[list[tuple[str, str]]],
    include_page_markers: bool = True,
) -> str:
    """Combine per-page, per-region extractions into a single markdown document.

    Input: page_extractions[page_idx] = [(region_label, extracted_content), ...]
    (Regions already in reading order.)
    Output: one markdown string.

    Formatting rules:
      - `text` regions: emitted as-is (preserving newlines)
      - `table` regions: emitted as-is (already HTML)
      - `figure` / `image` regions: emitted as `> Figure: <description>`
      - `seal` / `logo` regions: emitted as `> Seal: <identification>`
      - `chart` regions: emitted as-is (already markdown table)
      - Page separators use `[page_number]N[/page_number]` markers matching
        the reference output.md format.
    """
    out: list[str] = []
    for i, page in enumerate(page_extractions, start=1):
        if include_page_markers and i > 1:
            out.append(f"\n[page_number]{i}[/page_number]\n")
        for label, content in page:
            label_l = label.lower()
            content = (content or "").strip()
            if not content:
                continue
            if label_l in ("figure", "image", "picture"):
                out.append(f"\n> Figure: {content}\n")
            elif label_l in ("seal", "logo"):
                out.append(f"\n> Seal: {content}\n")
            else:
                # text, table, chart — emit as-is with blank-line separators
                out.append(content)
        out.append("")  # blank line between pages
    return "\n".join(out).strip() + "\n"


# ─── Validation (light-touch for MVP) ───────────────────────────────────────

def validate_output(markdown: str, source_pages: list[Image.Image]) -> dict:
    """Very light validation of assembled markdown.

    Returns a dict with issues found. Non-empty issues list = orchestrator
    may want to retry specific pages/regions. Full validation (compare against
    known page structure, verify table cell counts, detect language mismatches)
    is future work.
    """
    issues: list[str] = []
    if not markdown.strip():
        issues.append("empty_output")
    n_pages = len(source_pages)
    n_page_markers = markdown.count("[page_number]")
    if n_pages > 1 and n_page_markers < n_pages - 1:
        issues.append(
            f"missing_page_markers: expected {n_pages - 1}, found {n_page_markers}"
        )
    return {
        "n_chars":     len(markdown),
        "n_pages_in":  n_pages,
        "issues":      issues,
    }
