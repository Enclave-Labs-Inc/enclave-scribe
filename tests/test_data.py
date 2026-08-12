import json
from pathlib import Path

from PIL import Image

from scribe.data.dataset import DocumentDataset


def test_dataset_loads(tmp_path: Path):
    img = tmp_path / "sample.png"
    Image.new("RGB", (64, 64), color=255).save(img)

    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(json.dumps({"image": "sample.png", "text": "hello world"}) + "\n")

    ds = DocumentDataset(jsonl_path=str(jsonl), image_root=str(tmp_path))
    assert len(ds) == 1
    item = ds[0]
    assert item["text"] == "hello world"
    assert item["image"].size == (64, 64)


def test_dataset_empty_jsonl(tmp_path: Path):
    jsonl = tmp_path / "empty.jsonl"
    jsonl.write_text("")
    ds = DocumentDataset(jsonl_path=str(jsonl))
    assert len(ds) == 0
