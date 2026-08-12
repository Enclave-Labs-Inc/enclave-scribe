from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract interface every EnclaveScribe backend must implement."""

    @abstractmethod
    def load(self, model_path: str, **kwargs) -> None: ...

    @abstractmethod
    def infer(self, image_path: str, prompt: str, **kwargs) -> str:
        """Run OCR on a single image, return raw model output."""

    @abstractmethod
    def infer_batch(self, image_paths: list[str], prompt: str, **kwargs) -> list[str]:
        """Run OCR on multiple images, return list of raw outputs."""
