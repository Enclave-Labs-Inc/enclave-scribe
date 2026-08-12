from ..data.convert import to_images
from ..model.vlm import Qwen2VLModel
from ..postprocess.parser import clean_output


def infer_image(model: Qwen2VLModel, image_path: str, prompt: str = "document parsing.") -> str:
    return clean_output(model.infer(image_path, prompt))


def infer_document(
    model: Qwen2VLModel,
    file_path: str,
    prompt: str = "document parsing.",
    dpi: int = 300,
) -> list[str]:
    """Run OCR on any supported document type (image, PDF, PPT, DOCX)."""
    return [infer_image(model, page, prompt) for page in to_images(file_path, dpi=dpi)]
