"""Tools for the agentic OCR pipeline.

Simplified 2026-08-27: dropped per-region extractors (extract_table,
describe_image, identify_seal, extract_chart_data) and the router. One
page-level VLM call handles everything. See prompts.py for the rationale.

Public API:
    pdf_to_pages(path, dpi) -> list[PIL.Image]
    extract_page(image, registry) -> str (markdown)
    assemble_markdown(per_page_markdown, include_page_markers=True) -> str
    validate_output(markdown, source_pages) -> dict
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from . import prompts
from .models import ModelRegistry, VLMHandle


# ─── PDF → images ────────────────────────────────────────────────────────────

def pdf_to_pages(pdf_path: str | Path, dpi: int = 200) -> list[Image.Image]:
    """Convert a PDF to one PIL Image per page.

    Uses PyMuPDF (no external system deps like poppler). DPI 200 is a
    decent tradeoff: readable small text, keeps pages under ~5 MB.
    """
    import pymupdf

    doc = pymupdf.open(str(pdf_path))
    pages: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        pages.append(img)
    doc.close()
    return pages


# ─── VLM extraction ──────────────────────────────────────────────────────────

def extract_page(image: Image.Image, registry: ModelRegistry,
                 max_new_tokens: int = 4096) -> str:
    """Extract a whole page as markdown via one VLM call.

    The model is expected to handle text, tables, images, seals, and layout
    internally — that's what fine-tuning is for. Do not add per-region
    routing here.
    """
    import torch

    handle: VLMHandle = registry.vlm()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompts.PAGE},
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


# ─── Assembly ────────────────────────────────────────────────────────────────

def assemble_markdown(per_page_markdown: list[str],
                      include_page_markers: bool = True) -> str:
    """Concatenate per-page markdown into a single document.

    Emits `[page_number]N[/page_number]` separators matching the LlamaParse
    reference format. First page has no leading marker.
    """
    parts: list[str] = []
    for i, page_md in enumerate(per_page_markdown, start=1):
        if include_page_markers and i > 1:
            parts.append(f"\n[page_number]{i}[/page_number]\n")
        parts.append((page_md or "").strip())
        parts.append("")   # blank line between pages
    return "\n".join(parts).strip() + "\n"


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_output(markdown: str, source_pages: list[Image.Image]) -> dict:
    """Cheap sanity checks on the assembled markdown.

    Non-empty issues list = orchestrator may want to retry specific pages.
    Confidence-based retry loops are future work.
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
        "n_chars":    len(markdown),
        "n_pages_in": n_pages,
        "issues":     issues,
    }
