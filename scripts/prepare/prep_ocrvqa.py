"""OCR-VQA — VQA over book covers and product labels.

Dataset : howard-hou/OCR-VQA (~200k Q/A pairs over ~207k book/product images)
Output  : data/raw/ocrvqa/<idx>.png
          data/interim/ocrvqa.jsonl  (train pool)

This is the highest-signal source for iter-9's OCRBench V2 push — the
task shape (per-image VQA about textual content) is a direct analogue of
what OCRBench V2's `text_recognition` / `handwriting_ocr` / `key_information`
subsets grade.

Per-sample fields:
  image  → relative path under data/raw/
  text   → answer (first non-empty answer if multiple)
  prompt → question, formatted as "Question: {q}\\nAnswer:" for continuity
           with iter-7a's DocVQA prompt shape

The 2026-09-14 readiness audit confirmed `howard-hou/OCR-VQA` reachable (200 OK).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "howard-hou/OCR-VQA"


def _prompt_for(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 0) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID} ...")
    # OCR-VQA-200k is single-split; adjust if the schema on HF changes.
    ds = load_dataset(DATASET_ID, split="train")

    img_dir = raw_dir / "ocrvqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(tqdm(ds, desc="OCR-VQA")):
            if max_samples and n >= max_samples:
                break

            questions = sample.get("questions") or sample.get("question") or []
            answers = sample.get("answers") or sample.get("answer") or []
            if isinstance(questions, str):
                questions = [questions]
            if isinstance(answers, str):
                answers = [answers]
            if not questions or not answers:
                continue

            img_path = img_dir / f"{idx:07d}.png"
            if not img_path.exists():
                try:
                    sample["image"].save(img_path, "PNG")
                except Exception:
                    continue

            rel = str(img_path.relative_to(raw_dir))
            for q, a in zip(questions, answers):
                q_str = str(q).strip()
                a_str = str(a).strip()
                if not q_str or not a_str:
                    continue
                out_f.write(json.dumps({
                    "image":  rel,
                    "text":   a_str,
                    "prompt": _prompt_for(q_str),
                }, ensure_ascii=False) + "\n")
                n += 1
                if max_samples and n >= max_samples:
                    break

    print(f"OCR-VQA: {n:,} samples → {out_jsonl}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/ocrvqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Cap samples emitted (0 = unlimited).")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
