"""OmniDocBench — the primary benchmark for document OCR quality.

Dataset : opendatalab/OmniDocBench (HuggingFace)
Output  : data/raw/omnidocbench/<category>/<idx>.png
          data/benchmark/omnidocbench.jsonl

Categories (from the paper):
  academic, financial, handwritten, exam, magazine, newspaper,
  note, ppt, qa, textbook, law, book, table, formula
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def run(raw_dir: Path, out_jsonl: Path) -> int:
    print("Downloading OmniDocBench...")
    try:
        ds = load_dataset("opendatalab/OmniDocBench", trust_remote_code=True)
    except Exception as e:
        raise SystemExit(
            f"Failed to load OmniDocBench: {e}\n"
            "Ensure you have accepted the dataset license on HuggingFace and run:\n"
            "  huggingface-cli login"
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    category_counts: dict[str, int] = {}

    split_name = "test" if "test" in ds else list(ds.keys())[0]
    data = ds[split_name]

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(data, desc="OmniDocBench")):
            category = sample.get("category", sample.get("doc_type", "unknown"))
            text = sample.get("markdown", sample.get("text", sample.get("gt", ""))).strip()
            if not text:
                continue

            img_dir = raw_dir / "omnidocbench" / str(category)
            img_dir.mkdir(parents=True, exist_ok=True)
            img_path = img_dir / f"{idx:05d}.png"

            image = sample.get("image") or sample.get("img")
            if image is not None and not img_path.exists():
                if hasattr(image, "save"):
                    image.save(img_path)
                else:
                    img_path.write_bytes(image)

            if not img_path.exists():
                continue

            rel = str(img_path.relative_to(raw_dir))
            f.write(json.dumps(
                {"image": rel, "text": text, "category": str(category)},
                ensure_ascii=False,
            ) + "\n")
            category_counts[str(category)] = category_counts.get(str(category), 0) + 1
            count += 1

    print(f"\nOmniDocBench: {count} samples → {out_jsonl}")
    print("Per category:")
    for cat, n in sorted(category_counts.items()):
        print(f"  {cat:<20} {n:>5}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/benchmark/omnidocbench.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
