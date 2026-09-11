"""OCRBench V2 — the primary competitive benchmark (VISION.md target > 70.7%).

Dataset : lmms-lab/OCRBench-v2 (HuggingFace, ungated, 10,000 samples)
Layout  : Proper HF parquet dataset. Each row has:
            - image     (PIL image, embedded as bytes)
            - question  (string; the per-sample prompt)
            - answers   (sequence[string]; multiple valid answers)
            - dataset_name (string)
            - type      (string; task category — TextRecognition, HandwrittenMathExpression, etc.)
            - id        (int32)

Output  : data/raw/ocrbench_v2/images/<id>.png
          data/benchmark/ocrbench_v2.jsonl

JSONL shape (matches scripts/eval.py's expected {image, text, category, prompt}):

    {
      "image":    "ocrbench_v2/images/00042.png",
      "text":     "5.3",                      # answers[0] — primary ground truth
      "answers":  ["5.3", "5.30", "5.3 g"],   # full acceptable set (for a smarter scorer)
      "prompt":   "What is the mass shown in the image?",
      "category": "ocrbench_v2_TextRecognition",
      "id":       42
    }

**IMPORTANT scorer caveat**: OCRBench V2 has task-specific scoring rules —
Chart parsing uses relaxed numeric match, VQA uses exact-set match, Math uses
symbolic equivalence, etc. `scripts/eval.py`'s CER/WER/BLEU metrics only give
a first-cut number that is comparable across our own adapters but is NOT
directly comparable to Interfaze / Gemini / GPT / Claude reported OCRBench V2
scores until we implement the task-aware scorer (their official eval script
at github.com/Yuliang-Liu/MultimodalOCR — vendored in a future iter).

For iter-6's baseline measurement, the honest interpretation is:
  - CER/NED against `text` (= answers[0]) is an internal comparison signal
    across our own adapter checkpoints.
  - The published OCRBench V2 accuracy (70.7% for Interfaze) is NOT reproducible
    from our JSONL alone. Iter-6 report will state this explicitly.
  - OmniDocBench is the cleaner benchmark for headline "vs published models"
    because its scorer is edit-distance-based (NED) end-to-end.
"""
import argparse
import io
import json
from pathlib import Path

from tqdm import tqdm


REPO_ID = "lmms-lab/OCRBench-v2"


def run(raw_dir: Path, benchmark_jsonl: Path, limit: int = 0) -> int:
    from datasets import load_dataset

    img_dir = raw_dir / "ocrbench_v2" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {REPO_ID} test split (10,000 samples, ~5 GB — cached after first run)")
    ds = load_dataset(REPO_ID, split="test")
    total = len(ds) if limit == 0 else min(limit, len(ds))
    print(f"Emitting {total} samples → {benchmark_jsonl}")

    category_counts: dict[str, int] = {}
    dropped_no_answers = 0
    dropped_no_image = 0

    with open(benchmark_jsonl, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(tqdm(ds.select(range(total)), desc="OCRBench V2", total=total)):
            answers = row.get("answers") or []
            if not answers:
                dropped_no_answers += 1
                continue

            img = row.get("image")
            if img is None:
                dropped_no_image += 1
                continue

            sample_id = row.get("id", i)
            img_name = f"{sample_id:05d}.png"
            img_path = img_dir / img_name
            if not img_path.exists():
                # ds returns PIL.Image for image-dtype columns
                img.convert("RGB").save(img_path, format="PNG")

            task_type = str(row.get("type", "unknown")).strip() or "unknown"
            category = f"ocrbench_v2_{task_type.replace(' ', '_')}"
            category_counts[task_type] = category_counts.get(task_type, 0) + 1

            record = {
                "image":    str(img_path.relative_to(raw_dir)),
                "text":     answers[0].strip(),
                "answers":  [a.strip() for a in answers if a],
                "prompt":   (row.get("question") or "").strip() or "Read the text from the image.",
                "category": category,
                "id":       int(sample_id),
                "dataset_name": row.get("dataset_name", ""),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nOCRBench V2: {sum(category_counts.values())} samples → {benchmark_jsonl}")
    print(f"  Dropped (no answers): {dropped_no_answers}")
    print(f"  Dropped (no image):   {dropped_no_image}")
    print(f"  Per task type:")
    for t, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<40} {n:>5}")
    return sum(category_counts.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--benchmark_jsonl", default="data/benchmark/ocrbench_v2.jsonl")
    parser.add_argument("--limit",           type=int, default=0,
                        help="Debug: emit only first N samples (0 = all 10,000)")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.benchmark_jsonl), limit=args.limit)


if __name__ == "__main__":
    main()
