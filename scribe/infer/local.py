from ..data.pdf import pdf_to_images
from ..model.vlm import Qwen2VLModel
from ..postprocess.parser import clean_output


def infer_image(model: Qwen2VLModel, image_path: str, prompt: str = "document parsing.") -> str:
    return clean_output(model.infer(image_path, prompt))


def infer_pdf(
    model: Qwen2VLModel,
    pdf_path: str,
    prompt: str = "document parsing.",
    dpi: int = 300,
) -> list[str]:
    return [infer_image(model, page, prompt) for page in pdf_to_images(pdf_path, dpi=dpi)]
