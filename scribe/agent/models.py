"""Lazy-loaded model registry for the agent.

Two models needed for the MVP:
  - VLM (Qwen2.5-VL-7B-Instruct) — handles text/table/figure/seal extraction
  - Layout detector — finds regions and labels them

Both models are loaded ON FIRST USE (not import time) so importing this module
is cheap and CLI startup stays fast.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMHandle:
    """Wraps a loaded Qwen2.5-VL model + processor."""
    model: object
    processor: object
    device: str


class ModelRegistry:
    """Holds singleton model instances. Load once, reuse across tool calls."""

    def __init__(
        self,
        vlm_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_adapter_dir: str = "",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,  # higher than training cap — inference can afford it
    ):
        self.vlm_path = vlm_path
        self.vlm_adapter_dir = vlm_adapter_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._vlm: Optional[VLMHandle] = None
        self._layout = None

    def vlm(self) -> VLMHandle:
        """Return a loaded Qwen2.5-VL VLM handle. Loads on first call."""
        if self._vlm is not None:
            return self._vlm

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Pick the best available device (CUDA > MPS > CPU).
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        processor = AutoProcessor.from_pretrained(
            self.vlm_path,
            trust_remote_code=True,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            self.vlm_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            attn_implementation="sdpa",
            device_map=device if device != "cpu" else None,
            trust_remote_code=True,
        )

        if self.vlm_adapter_dir:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.vlm_adapter_dir)

        model = model.eval()
        self._vlm = VLMHandle(model=model, processor=processor, device=device)
        return self._vlm

    def layout(self):
        """Return a loaded layout detector.

        MVP: returns a stub detector that treats each page as one 'text' region.
        Swap this for DocLayout-YOLO or similar in a follow-up PR — the
        detector interface is `detect(image) -> list[Region]`.
        """
        if self._layout is not None:
            return self._layout
        self._layout = _StubLayoutDetector()
        return self._layout


class _StubLayoutDetector:
    """Placeholder layout detector — returns the whole page as one text region.

    Real implementation (follow-up): use DocLayout-YOLO to detect text blocks,
    tables, figures, and seals with bounding boxes. Same output shape.
    """

    def detect(self, image):
        w, h = image.size
        return [
            {
                "label": "text",
                "bbox": (0, 0, w, h),
                "confidence": 1.0,
            }
        ]
