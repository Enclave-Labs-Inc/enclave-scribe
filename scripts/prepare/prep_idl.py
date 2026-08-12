"""IDL-WDS — real industry document pages (tobacco, pharma, legal).

Dataset : cathyzheng/IDL-WDS on HuggingFace (~6M pages, WebDataset format)
          We sample a configurable subset (default 250K) for training.
Output  : data/raw/idl/<shard_idx>/<page_idx>.png
          data/interim/idl.jsonl

Note    : Each page has OCR text extracted from the original PDF.
          Text quality varies — real documents, real OCR noise.
"""
import argparse
import io
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 250_000) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading IDL-WDS (streaming, max {max_samples:,} samples)...")
    ds = load_dataset(
        "cathyzheng/IDL-WDS",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    img_base = raw_dir / "idl"
    img_base.mkdir(parents=True, exist_ok=True)
    count = 0
    shard = 0
    shard_dir = img_base / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for sample in tqdm(ds, total=max_samples, desc="IDL-WDS"):
            if count >= max_samples:
                break

            if count > 0 and count % 5000 == 0:
                shard += 1
                shard_dir = img_base / f"{shard:04d}"
                shard_dir.mkdir(exist_ok=True)

            text = sample.get("txt", "").strip()
            if not text or len(text) < 20:
                continue

            img_data = sample.get("png") or sample.get("jpg") or sample.get("image")
            if img_data is None:
                continue

            try:
                if isinstance(img_data, bytes):
                    image = Image.open(io.BytesIO(img_data)).convert("RGB")
                elif hasattr(img_data, "convert"):
                    image = img_data.convert("RGB")
                else:
                    continue
            except Exception:
                continue

            img_path = shard_dir / f"{count % 5000:05d}.png"
            image.save(img_path)

            rel = str(img_path.relative_to(raw_dir))
            out_f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
            count += 1

    print(f"IDL-WDS: {count:,} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/idl.jsonl")
    parser.add_argument("--max_samples", type=int, default=250_000)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
