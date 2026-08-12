import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from .base import BaseModel


class Qwen2VLModel(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None

    def load(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", **kwargs) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **kwargs,
        ).eval()

    def _build_messages(self, image: Image.Image, prompt: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def infer(self, image_path: str, prompt: str = "document parsing.", **kwargs) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = self._build_messages(image, prompt)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 4096),
                do_sample=False,
            )
        return self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]

    def infer_batch(self, image_paths: list[str], prompt: str = "document parsing.", **kwargs) -> list[str]:
        return [self.infer(p, prompt, **kwargs) for p in image_paths]
