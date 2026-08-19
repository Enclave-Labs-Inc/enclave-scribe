from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentCollator:
    processor: Any
    prompt: str = "document parsing."
    max_length: int = 4096

    def __call__(self, batch: list[dict]) -> dict:
        messages_list, images_list = [], []
        for item in batch:
            images_list.append(item["image"])
            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["text"]}],
                },
            ])

        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in messages_list
        ]
        return self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
