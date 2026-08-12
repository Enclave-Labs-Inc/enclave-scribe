import json
import os

import requests

from ..data.convert import to_images
from ..utils.image import encode_image_b64

_DEFAULT_URL = "http://127.0.0.1:10000"
_MODEL_NAME = "enclave-scribe"


def _build_payload(image_path: str, prompt: str) -> dict:
    mime, data = encode_image_b64(image_path)
    return {
        "model": _MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }
        ],
        "temperature": 0,
        "stream": True,
    }


def infer_image(
    image_path: str,
    prompt: str = "document parsing.",
    server_url: str = _DEFAULT_URL,
    timeout: int = 1200,
) -> str:
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(_build_payload(image_path, prompt)),
        timeout=timeout,
        stream=True,
    )
    resp.raise_for_status()
    chunks = []
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode() if isinstance(raw, bytes) else raw
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            if delta:
                chunks.append(delta)
        except (json.JSONDecodeError, KeyError):
            continue
    return "".join(chunks)


def infer_document(file_path: str, dpi: int = 300, **kwargs) -> list[str]:
    """Run OCR on any supported document type (image, PDF, PPT, DOCX) via the SGLang server."""
    return [infer_image(page, **kwargs) for page in to_images(file_path, dpi=dpi)]
