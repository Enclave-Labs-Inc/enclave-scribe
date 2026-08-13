"""OpenAI-compatible client for the SGLang backend (port 10000).

SGLang supports a custom n-gram logit processor to suppress OCR repetition loops.
Pass `ngram_ban=True` (default) to activate it via the extra_body field.
"""
import json

import requests

from ..data.convert import to_images
from ..utils.image import encode_image_b64

_DEFAULT_URL = "http://127.0.0.1:10000"
_MODEL_NAME = "enclave-scribe"


def _build_payload(
    image_path: str,
    prompt: str,
    max_tokens: int = 4096,
    ngram_ban: bool = True,
    ngram_size: int = 8,
    ngram_window: int = 512,
) -> dict:
    mime, data = encode_image_b64(image_path)
    payload: dict = {
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
        "max_tokens": max_tokens,
        "stream": True,
    }
    if ngram_ban:
        payload["extra_body"] = {
            "custom_logit_processor": {
                "type": "ngram_ban",
                "ngram_size": ngram_size,
                "window": ngram_window,
            }
        }
    return payload


def infer_image(
    image_path: str,
    prompt: str = "document parsing.",
    server_url: str = _DEFAULT_URL,
    max_tokens: int = 4096,
    ngram_ban: bool = True,
    ngram_size: int = 8,
    ngram_window: int = 512,
    timeout: int = 1200,
) -> str:
    resp = requests.post(
        f"{server_url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            _build_payload(image_path, prompt, max_tokens, ngram_ban, ngram_size, ngram_window)
        ),
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
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
            if delta:
                chunks.append(delta)
        except (json.JSONDecodeError, KeyError):
            continue
    return "".join(chunks)


def infer_document(file_path: str, dpi: int = 300, **kwargs) -> list[str]:
    """Run OCR on any supported document type via SGLang."""
    return [infer_image(page, **kwargs) for page in to_images(file_path, dpi=dpi)]
