"""TextVQA — VQA over natural images containing scene text.

Dataset : lmms-lab/textvqa (public mirror of the original textvqa/textvqa)
Output  : data/raw/textvqa/<split>_<idx>.png
          data/interim/textvqa.jsonl

Iter-9 uses `lmms-lab/textvqa` because the canonical `textvqa/textvqa`
repo is gated (401 for anonymous + non-approved tokens). The 2026-09-14
readiness audit confirmed `lmms-lab/textvqa` as the working alternate.

Scene-text VQA complements OCR-VQA (books/products) with in-the-wild
text — signs, packaging, natural documents. OCRBench V2 has a
`text_recognition` subset that tests exactly this.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "lmms-lab/textvqa"


def _prompt_for(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _pick_answer(sample: dict) -> str:
    """TextVQA has 10 crowd answers per question; pick the most common non-empty."""
    from collections import Counter
    raw = sample.get("answers") or []
    cleaned = [str(a).strip() for a in raw if str(a).strip()]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    return counts.most_common(1)[0][0]


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 0) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)

    img_dir = raw_dir / "textvqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for split_name in ("train", "validation", "val"):
            if split_name not in ds:
                continue
            for idx, sample in enumerate(tqdm(ds[split_name], desc=f"TextVQA/{split_name}")):
                if max_samples and n >= max_samples:
                    break
                question = str(sample.get("question") or "").strip()
                answer = _pick_answer(sample)
                if not question or not answer:
                    continue

                img_path = img_dir / f"{split_name}_{idx:06d}.png"
                if not img_path.exists():
                    try:
                        sample["image"].save(img_path, "PNG")
                    except Exception:
                        continue

                rel = str(img_path.relative_to(raw_dir))
                out_f.write(json.dumps({
                    "image":  rel,
                    "text":   answer,
                    "prompt": _prompt_for(question),
                }, ensure_ascii=False) + "\n")
                n += 1
            if max_samples and n >= max_samples:
                break

    print(f"TextVQA: {n:,} samples → {out_jsonl}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/textvqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
