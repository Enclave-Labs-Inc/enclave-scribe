"""Lazy-loaded VLM registry for the agent.

Simplified 2026-08-27: dropped the stubbed layout detector. Page-level
extraction doesn't need layout detection — the VLM handles the whole
page in one shot.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class VLMHandle:
    """Wraps a loaded Qwen2.5-VL (or OLMoCR fine-tune) model + processor."""
    model: object
    processor: object
    device: str


class ModelRegistry:
    """Holds the singleton VLM instance. Load once, reuse across calls."""

    def __init__(
        self,
        vlm_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_adapter_dir: str = "",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        self.vlm_path = vlm_path
        self.vlm_adapter_dir = vlm_adapter_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._vlm: Optional[VLMHandle] = None

    def vlm(self) -> VLMHandle:
        """Return a loaded VLM handle. Loads on first call."""
        if self._vlm is not None:
            return self._vlm

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

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

        self._vlm = VLMHandle(model=model.eval(), processor=processor, device=device)
        return self._vlm
