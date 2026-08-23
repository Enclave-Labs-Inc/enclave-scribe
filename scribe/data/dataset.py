import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .transforms import apply_train_transforms


class DocumentDataset(Dataset):
    """
    JSONL dataset where each line is:
      {"image": "relative/path.png", "text": "ground truth markdown",
       "prompt": "optional per-sample prompt override"}

    When a sample has no `prompt` field, the collator's default prompt applies.
    Per-sample prompts let us mix task types in one training run
    (e.g. "document parsing." for OCR datasets, "Question: X\\nAnswer:" for DocVQA).
    """

    def __init__(self, jsonl_path: str, image_root: str = "", augment: bool = False):
        self.image_root = Path(image_root) if image_root else None
        self.augment = augment
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = (self.image_root / sample["image"]) if self.image_root else Path(sample["image"])
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = apply_train_transforms(image)
        item = {"image": image, "text": sample["text"], "image_path": str(image_path)}
        if "prompt" in sample:
            item["prompt"] = sample["prompt"]
        return item


class UnslothDocumentDataset(Dataset):
    """
    Same JSONL format but returns Unsloth conversation dicts.
    Unsloth's SFTTrainer + UnslothVisionDataCollator expects this shape.
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: str = "",
        prompt: str = "document parsing.",
        augment: bool = False,
    ):
        self.image_root = Path(image_root) if image_root else None
        self.prompt = prompt
        self.augment = augment
        self.samples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = (self.image_root / sample["image"]) if self.image_root else Path(sample["image"])
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = apply_train_transforms(image)

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",  "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["text"]}],
                },
            ]
        }
