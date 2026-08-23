"""DocVQA — real scanned documents with question/answer supervision.

Dataset : HuggingFaceM4/DocumentVQA  (~50k samples, public)
Output  : data/raw/docvqa/<split>/<idx>.png
          data/interim/docvqa.jsonl                (train split → training pool)
          data/benchmark/docvqa_test.jsonl         (val split  → held-out)

This is a Q&A dataset, not pure OCR. Each sample emits a `prompt` field
so the training collator can switch its input prompt per-sample (see
follow-up PR that adds prompt-per-dataset support). Without that support,
the extra field is silently ignored and the default prompt applies.

Prompt format used:
  "Question: {question}\\nAnswer:"
Target text is the first accepted answer from the `answers` list.

Question types (used as category tag on held-out samples):
  handwritten, form, layout, table/list, free_text, others
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "HuggingFaceM4/DocumentVQA"


def _prompt_for(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def _category_tag(question_types) -> str:
    if not question_types:
        return "docvqa_other"
    first = str(question_types[0]).strip().lower().replace("/", "_").replace(" ", "_")
    return f"docvqa_{first}"


def run(raw_dir: Path, out_jsonl: Path, benchmark_jsonl: Path | None = None) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID)  # train / validation / test

    train_count = 0
    test_count = 0
    with open(out_jsonl, "w", encoding="utf-8") as train_f, \
         open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else _null_writer() as test_f:

        # train split → training pool
        img_dir = raw_dir / "docvqa" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        for idx, sample in enumerate(tqdm(ds["train"], desc="DocVQA/train")):
            answers = sample.get("answers", []) or []
            if not answers:
                continue
            answer = str(answers[0]).strip()
            if not answer:
                continue

            img_path = img_dir / f"{idx:06d}.png"
            if not img_path.exists():
                sample["image"].save(img_path, "PNG")

            rel = str(img_path.relative_to(raw_dir))
            train_f.write(json.dumps({
                "image":  rel,
                "text":   answer,
                "prompt": _prompt_for(sample["question"]),
            }, ensure_ascii=False) + "\n")
            train_count += 1

        # validation split → held-out benchmark (if configured)
        if benchmark_jsonl is not None:
            img_dir = raw_dir / "docvqa" / "validation"
            img_dir.mkdir(parents=True, exist_ok=True)
            for idx, sample in enumerate(tqdm(ds["validation"], desc="DocVQA/val")):
                answers = sample.get("answers", []) or []
                if not answers:
                    continue
                answer = str(answers[0]).strip()
                if not answer:
                    continue

                img_path = img_dir / f"{idx:06d}.png"
                if not img_path.exists():
                    sample["image"].save(img_path, "PNG")

                rel = str(img_path.relative_to(raw_dir))
                test_f.write(json.dumps({
                    "image":    rel,
                    "text":     answer,
                    "prompt":   _prompt_for(sample["question"]),
                    "category": _category_tag(sample.get("question_types", [])),
                }, ensure_ascii=False) + "\n")
                test_count += 1

    print(f"DocVQA: {train_count:,} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"DocVQA: {test_count:,} val (held out) → {benchmark_jsonl}")
    return train_count + test_count


class _null_writer:
    """No-op file object stand-in for when benchmark_jsonl is not provided."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def write(self, _): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--out_jsonl",       default="data/interim/docvqa.jsonl")
    parser.add_argument("--benchmark_jsonl", default="",
                        help="If set, validation split is routed here as held-out test set")
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(Path(args.raw_dir), Path(args.out_jsonl), bench)


if __name__ == "__main__":
    main()
