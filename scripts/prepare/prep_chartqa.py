"""ChartQA — VQA over charts and graphs.

Dataset : ahmed-masry/ChartQA (~28k Q/A pairs over ~20k chart images,
          human + augmented splits)
Output  : data/raw/chartqa/<idx>.png
          data/interim/chartqa.jsonl  (train pool, human + augmented)

Chart understanding is a direct OCRBench V2 subset. This corpus + iter-7a's
DocVQA baseline teaches the model to answer numeric/categorical
questions grounded in extracted chart values.

The 2026-09-14 readiness audit confirmed `ahmed-masry/ChartQA` reachable (200 OK).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "ahmed-masry/ChartQA"


def _prompt_for(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 0) -> int:
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
                # ChartQA labels the answer as `label` (single value), sometimes `answer`.
                answer_field = sample.get("label") or sample.get("answer") or ""
                if isinstance(answer_field, list):
                    answer = str(answer_field[0]).strip() if answer_field else ""
                else:
                    answer = str(answer_field).strip()
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

    print(f"ChartQA: {n:,} samples → {out_jsonl}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/chartqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
