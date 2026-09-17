"""ChartQA — VQA over charts and graphs.

Dataset : ahmed-masry/ChartQA (~28k Q/A pairs over ~20k chart images,
          human + augmented splits)
Output  : data/raw/chartqa/<idx>.png
          data/interim/chartqa.jsonl  (train pool, human + augmented)

Chart understanding is a direct OCRBench V2 subset. This corpus + iter-7a's
DocVQA baseline teaches the model to answer numeric/categorical
questions grounded in extracted chart values.

Schema (confirmed 2026-09-17):
  imgname (str), query (str), label (str), type (str),
  image (binary bytes — NOT a PIL Image; must be decoded via PIL.Image.open)

WHY THIS REWRITE (iter-12)
    Iter-11's version silently returned 0 samples. `sample["image"]` is raw
    bytes, so `.save()` raised `AttributeError` on every row — swallowed by
    the broad `except Exception`. Fix: decode bytes with
    `PIL.Image.open(BytesIO(...))` before saving.

# mirror integration deferred: mirror strips binary bytes; see mirror_datasets_to_s3.py
"""
from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


DATASET_ID = "ahmed-masry/ChartQA"

PROMPT = "Answer this chart question."


def _prompt_for(question: str) -> str:
    return f"{PROMPT}\nQuestion: {question.strip()}\nAnswer:"


def _decode_image(raw) -> Image.Image | None:
    """Turn the `image` field into a PIL Image regardless of its wire shape."""
    if raw is None:
        return None
    if isinstance(raw, Image.Image):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return Image.open(BytesIO(raw))
        except (OSError, ValueError, UnidentifiedImageError) as e:
            print(f"chartqa decode-bytes: {e}", file=sys.stderr)
            return None
    # datasets occasionally wraps bytes in a dict: {"bytes": ..., "path": ...}
    if isinstance(raw, dict) and raw.get("bytes"):
        try:
            return Image.open(BytesIO(raw["bytes"]))
        except (OSError, ValueError, UnidentifiedImageError, KeyError) as e:
            print(f"chartqa decode-dict-bytes: {e}", file=sys.stderr)
            return None
    return None


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 5000) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    img_dir = raw_dir / "chartqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for split_name in ("train", "val", "validation"):
            if split_name not in ds:
                continue
            for idx, sample in enumerate(tqdm(ds[split_name], desc=f"ChartQA/{split_name}")):
                if max_samples and n >= max_samples:
                    break
                question = str(sample.get("query") or sample.get("question") or "").strip()
                answer_field = sample.get("label") or sample.get("answer") or ""
                if isinstance(answer_field, list):
                    answer = str(answer_field[0]).strip() if answer_field else ""
                else:
                    answer = str(answer_field).strip()
                if not question or not answer:
                    continue

                img = _decode_image(sample.get("image"))
                if img is None:
                    continue

                img_path = img_dir / f"{split_name}_{idx:06d}.png"
                if not img_path.exists():
                    try:
                        img.save(img_path, "PNG")
                    except (OSError, ValueError) as e:
                        print(f"chartqa save {img_path.name}: {e}", file=sys.stderr)
                        continue

                rel = str(img_path.relative_to(raw_dir))
                out_f.write(json.dumps({
                    "image":  rel,
                    "text":   answer,
                    "prompt": _prompt_for(question),
                    "source": "chartqa",
                }, ensure_ascii=False) + "\n")
                n += 1
            if max_samples and n >= max_samples:
                break

    print(f"ChartQA: {n:,} samples → {out_jsonl}")

    if n == 0:
        raise RuntimeError("chartqa produced 0 samples")

    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/chartqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=5000)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
