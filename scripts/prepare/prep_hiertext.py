"""HierText — real natural images with hierarchical paragraph/line/word text.

Dataset : Google Research, CC-BY 4.0
Images  : Open Images subset (~11.6K train, ~1.3K val)
Source  : https://github.com/google-research-datasets/hiertext
Output  : data/raw/hiertext/<split>/<image_id>.jpg
          data/interim/hiertext.jsonl

Requires: pip install fiftyone  (for Open Images download)
          OR manually place images in data/raw/hiertext/<split>/
"""
import argparse
import json
import os
from pathlib import Path

import requests
from tqdm import tqdm

GT_URLS = {
    "train": "https://storage.googleapis.com/hiertext/hiertext/train.jsonl.gz",
    "validation": "https://storage.googleapis.com/hiertext/hiertext/validation.jsonl.gz",
}
IMAGE_BASE = "https://storage.googleapis.com/openimages/data/validation/"


def _download(url: str, dest: Path):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(8192):
            f.write(chunk)
            bar.update(len(chunk))


def _download_image(image_id: str, dest: Path) -> bool:
    if dest.exists():
        return True
    url = f"https://storage.googleapis.com/openimages/data/train/{image_id}.jpg"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def _annotations_to_text(annotations: dict) -> str:
    paragraphs = []
    for para in annotations.get("paragraphs", []):
        if not para.get("legible", True):
            continue
        lines_text = []
        for line in para.get("lines", []):
            if not line.get("legible", True):
                continue
            words = [
                w["text"] for w in line.get("words", [])
                if w.get("legible", True) and w.get("text", "").strip()
            ]
            if words:
                lines_text.append(" ".join(words))
        if lines_text:
            paragraphs.append("\n".join(lines_text))
    return "\n\n".join(paragraphs)


def run(raw_dir: Path, out_jsonl: Path) -> int:
    import gzip

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for split, gt_url in GT_URLS.items():
            gt_gz = raw_dir / "hiertext" / f"{split}.jsonl.gz"
            _download(gt_url, gt_gz)

            img_dir = raw_dir / "hiertext" / split
            img_dir.mkdir(parents=True, exist_ok=True)

            with gzip.open(gt_gz, "rt", encoding="utf-8") as gz_f:
                for line in tqdm(gz_f, desc=f"HierText/{split}"):
                    sample = json.loads(line)
                    image_id = sample["image_id"]
                    img_path = img_dir / f"{image_id}.jpg"

                    if not _download_image(image_id, img_path):
                        continue

                    text = _annotations_to_text(sample)
                    if not text.strip():
                        continue

                    rel = str(img_path.relative_to(raw_dir))
                    out_f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                    count += 1

    print(f"HierText: {count} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/hiertext.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
