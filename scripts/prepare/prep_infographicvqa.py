"""InfographicVQA — VQA over dense infographics with mixed text + graphics.

Dataset : HuggingFaceM4/InfographicVQA (mirror of the original vqa-infographic
          collection which is 401-gated).
Output  : data/raw/infographicvqa/<split>_<idx>.png
          data/interim/infographicvqa.jsonl

Infographics are the closest single-image task shape to OCRBench V2's
"document understanding" subset — complex layouts, mixed graphics and
text, questions that require reading numbers/labels in context.

The 2026-09-14 readiness audit noted `vqa-infographic/infographicvqa`
returns 401. The corpus builder tries `HuggingFaceM4/InfographicVQA`
first; if that also 401s at runtime, the source is DROPPED from the mix
(5k / 4% — droppable per plan) and remaining sources compensate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "HuggingFaceM4/InfographicVQA"


def _prompt_for(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _pick_answer(sample: dict) -> str:
    ans = sample.get("answers") or sample.get("answer") or []
    if isinstance(ans, str):
        return ans.strip()
    for a in ans:
        s = str(a).strip()
        if s:
            return s
    return ""


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 0) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {DATASET_ID} ...")
    try:
        ds = load_dataset(DATASET_ID)
    except Exception as e:
        print(f"WARN: {DATASET_ID} failed to load ({e}). Iter-9 will proceed without InfographicVQA — 5k/120k = 4% of corpus, per plan droppable.")
        # Still emit an empty JSONL so the corpus builder's idempotent skip works.
        out_jsonl.write_text("", encoding="utf-8")
        return 0

    img_dir = raw_dir / "infographicvqa"
    img_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for split_name in ("train", "validation", "val"):
            if split_name not in ds:
                continue
            for idx, sample in enumerate(tqdm(ds[split_name], desc=f"InfographicVQA/{split_name}")):
                if max_samples and n >= max_samples:
                    break
                question = str(sample.get("question") or sample.get("query") or "").strip()
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

    print(f"InfographicVQA: {n:,} samples → {out_jsonl}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/infographicvqa.jsonl")
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
