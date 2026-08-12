import base64
import io
import os
import tempfile

from fastapi import APIRouter, HTTPException
from PIL import Image

from scribe.postprocess.parser import clean_output
from scribe.utils.logging import get_logger
from ..schemas import OCRRequest, OCRResponse

router = APIRouter()
logger = get_logger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from scribe.model.vlm import Qwen2VLModel
        _model = Qwen2VLModel()
        _model.load(os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    return _model


@router.post("/v1/ocr", response_model=OCRResponse)
async def ocr(req: OCRRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            result = _get_model().infer(tmp_path, prompt=req.prompt)
        finally:
            os.unlink(tmp_path)

        if req.clean_output:
            result = clean_output(result)

        return OCRResponse(text=result, tokens=len(result.split()))
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
