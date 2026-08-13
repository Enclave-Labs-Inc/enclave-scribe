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

# SCRIBE_BACKEND: local | vllm | sglang  (default: local)
_BACKEND = os.getenv("SCRIBE_BACKEND", "local").lower()
_SERVER_URL = os.getenv("SCRIBE_SERVER_URL", "")

_local_model = None


def _get_local_model():
    global _local_model
    if _local_model is None:
        from scribe.model.vlm import Qwen2VLModel
        _local_model = Qwen2VLModel()
        _local_model.load(os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-7B-Instruct"))
    return _local_model


def _infer_server(image_path: str, prompt: str) -> str:
    if _BACKEND == "vllm":
        from scribe.infer.vllm_client import infer_image
        url = _SERVER_URL or "http://127.0.0.1:8000"
    else:  # sglang
        from scribe.infer.sglang_client import infer_image
        url = _SERVER_URL or "http://127.0.0.1:10000"
    return infer_image(image_path, prompt=prompt, server_url=url)


@router.post("/v1/ocr", response_model=OCRResponse)
async def ocr(req: OCRRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        try:
            if _BACKEND == "local":
                result = _get_local_model().infer(tmp_path, prompt=req.prompt)
            else:
                result = _infer_server(tmp_path, req.prompt)
        finally:
            os.unlink(tmp_path)

        if req.clean_output:
            result = clean_output(result)

        return OCRResponse(text=result, tokens=len(result.split()))
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
