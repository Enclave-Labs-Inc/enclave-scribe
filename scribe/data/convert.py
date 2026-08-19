"""Universal document-to-images converter.

Supported inputs:
  - Images  : PNG, JPG, JPEG, WEBP, BMP, TIFF
  - PDF     : via PyMuPDF
  - PPT/PPTX: via LibreOffice headless → PDF → images
  - DOC/DOCX: via LibreOffice headless → PDF → images
"""
import os
import subprocess
import tempfile
from pathlib import Path

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_PDF_EXTS = {".pdf"}
_OFFICE_EXTS = {".ppt", ".pptx", ".doc", ".docx", ".odp", ".odt"}

SUPPORTED_EXTS = _IMAGE_EXTS | _PDF_EXTS | _OFFICE_EXTS


def to_images(file_path: str, dpi: int = 300) -> list[str]:
    """Convert any supported document to a list of image paths."""
    ext = Path(file_path).suffix.lower()
    if ext in _IMAGE_EXTS:
        return [file_path]
    if ext in _PDF_EXTS:
        return _pdf_to_images(file_path, dpi)
    if ext in _OFFICE_EXTS:
        return _office_to_images(file_path, dpi)
    raise ValueError(f"Unsupported file type: {ext!r}. Supported: {sorted(SUPPORTED_EXTS)}")


def _pdf_to_images(pdf_path: str, dpi: int = 300) -> list[str]:
    import fitz

    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="scribe_pdf_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def _office_to_images(office_path: str, dpi: int = 300) -> list[str]:
    """Convert PPT/PPTX/DOC/DOCX to images via LibreOffice headless."""
    tmp_dir = tempfile.mkdtemp(prefix="scribe_office_")
    try:
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                "--outdir", tmp_dir,
                office_path,
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "LibreOffice is required for PPT/DOCX conversion. "
            "Install with: brew install libreoffice  (macOS)  or  apt install libreoffice  (Linux)"
        )
    pdf_path = os.path.join(tmp_dir, Path(office_path).stem + ".pdf")
    if not os.path.exists(pdf_path):
        raise RuntimeError(f"LibreOffice conversion failed — no PDF produced for {office_path}")
    return _pdf_to_images(pdf_path, dpi)
