"""FUNSD — real scanned English forms with text annotations.

Dataset : nielsr/funsd  (~200 samples)
Output  : data/raw/funsd/<split>/<idx>.png
          data/interim/funsd.jsonl
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _words_to_text(words: list[str], bboxes: list[list[int]]) -> str:
    if not words:
        return ""
    pairs = sorted(zip(words, bboxes), key=lambda p: (p[1][1], p[1][0]))
    return " ".join(w for w, _ in pairs if w.strip())


def run(raw_dir: Path, out_jsonl: Path, benchmark_jsonl: Path | None = None) -> int:
    """Route FUNSD splits: train → out_jsonl (training pool);
    test → benchmark_jsonl (held-out test set), if provided.
    Falls back to old behavior (both splits → out_jsonl) when benchmark_jsonl is None.
    """
    ds = load_dataset("nielsr/funsd")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    with open(out_jsonl, "w", encoding="utf-8") as train_f, \
         open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else _null_writer() as test_f:
        for split in ["train", "test"]:
            img_dir = raw_dir / "funsd" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            is_heldout = (split == "test" and benchmark_jsonl is not None)
            target = test_f if is_heldout else train_f

            for idx, sample in enumerate(tqdm(ds[split], desc=f"FUNSD/{split}")):
                text = _words_to_text(sample["words"], sample["bboxes"])
                if not text.strip():
                    continue

                img_path = img_dir / f"{idx:05d}.png"
                if not img_path.exists():
                    sample["image"].save(img_path)

                rel = str(img_path.relative_to(raw_dir))
                record = {"image": rel, "text": text}
                if is_heldout:
                    record["category"] = "funsd"
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                if is_heldout:
                    test_count += 1
                else:
                    train_count += 1

    print(f"FUNSD: {train_count} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"FUNSD: {test_count} test → {benchmark_jsonl} (held out)")
    return train_count + test_count


class _null_writer:
    """No-op file object stand-in for when benchmark_jsonl is not provided."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def write(self, _): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--out_jsonl",       default="data/interim/funsd.jsonl")
    parser.add_argument("--benchmark_jsonl", default="",
                        help="If set, official test split is routed here instead of --out_jsonl")
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(Path(args.raw_dir), Path(args.out_jsonl), bench)


if __name__ == "__main__":
    main()
