from ..data.convert import to_images
from ..model.vlm import Qwen2VLModel
from ..postprocess.parser import clean_output


def infer_image(model: Qwen2VLModel, image_path: str, prompt: str = "document parsing.", **kwargs) -> str:
    return clean_output(model.infer(image_path, prompt, **kwargs))


def infer_document(
    model: Qwen2VLModel,
    file_path: str,
    prompt: str = "document parsing.",
    dpi: int = 300,
    **kwargs,
) -> list[str]:
    """Run OCR on any supported document type (image, PDF, PPT, DOCX)."""
    return [infer_image(model, page, prompt, **kwargs) for page in to_images(file_path, dpi=dpi)]
