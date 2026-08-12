import base64
import io
import os

from PIL import Image


def encode_image_b64(image_path: str) -> tuple[str, str]:
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/jpeg" if ext in (".jpg", ".jpeg") else f"image/{ext.lstrip('.')}"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return mime, data


def decode_image_b64(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


def resize_max(image: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
