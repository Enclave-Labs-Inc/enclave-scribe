import io
import random

from PIL import Image, ImageFilter


def random_jpeg_compression(image: Image.Image, quality_range: tuple[int, int] = (70, 95)) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=random.randint(*quality_range))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_blur(image: Image.Image, radius_range: tuple[float, float] = (0.5, 1.5)) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(*radius_range)))


def apply_train_transforms(image: Image.Image, p: float = 0.3) -> Image.Image:
    if random.random() < p:
        image = random_jpeg_compression(image)
    if random.random() < p:
        image = random_blur(image)
    return image
