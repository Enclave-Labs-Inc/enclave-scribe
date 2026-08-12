import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from .base import BaseModel


def _bnb_config(bits: int = 4) -> BitsAndBytesConfig:
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unsupported quantization bits: {bits}. Use 4 or 8.")


class Qwen2VLModel(BaseModel):
    def __init__(self):
        self.model = None
        self.processor = None

    def load(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        quantize: int | None = None,   # 4 → QLoRA, 8 → int8, None → bf16
        **kwargs,
    ) -> None:
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        load_kwargs = dict(device_map="auto", **kwargs)
        if quantize:
            load_kwargs["quantization_config"] = _bnb_config(quantize)
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, **load_kwargs
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
        text = self.processor.apply_chat_template(
            self._build_messages(image, prompt), tokenize=False, add_generation_prompt=True
        )
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
