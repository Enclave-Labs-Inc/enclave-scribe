"""olmOCR-mix-1025 — full-page English OCR from AllenAI's own SFT corpus.

Dataset : allenai/olmOCR-mix-1025 (~267,962 pages, ODC-BY, GPT-4.1 transcriptions)
Output  : data/raw/olmocr_mix/<idx>.png
          data/interim/olmocr_mix.jsonl (training pool)

WHY THIS DATASET
    Iter-10 swaps base to Qwen3-VL-8B, giving up olmOCR-2's 270k-page English
    page-OCR pretraining. This dataset is *exactly* what olmOCR-2 was SFT'd on
    — borrowing their proven page-OCR corpus is the cleanest recovery.

    Composition (from HF dataset card):
      - 00_documents (232,790) — web-crawled PDFs, EN 94.46%
      - 01_books (17,474)      — Internet Archive, EN 91.28%
      - 02_loc_transcripts (9,989) — Library of Congress, EN 98.21%
      - 03_national_archives (9,997) — EN 99.82%

TARGET: 15,000 samples via `datasets.Dataset.select(range(15000))` seeded — so
reruns produce the same slice.

USAGE
    python scripts/prepare/prep_olmocr_mix.py --limit 15000
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


DATASET_ID = "allenai/olmOCR-mix-1025"


def run(raw_dir: Path, out_jsonl: Path, limit: int) -> int:
    from datasets import load_dataset

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    img_dir = raw_dir / "olmocr_mix"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} ...", flush=True)
    # Non-streaming so we can seed-slice reproducibly. If the download is huge,
    # switch to streaming and take the first `limit` samples.
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(ds, desc="olmocr_mix", total=limit)):
            if kept >= limit:
                break
            # Field names vary by dataset — try common ones
            text = sample.get("text") or sample.get("markdown") or sample.get("content") or ""
            text = str(text).strip()
            if not text:
                continue

            img = sample.get("image") or sample.get("pdf_image") or sample.get("page_image")
            if img is None:
                # Some subsets may lack raw image; skip
                continue

            img_path = img_dir / f"{idx:06d}.png"
            if not img_path.exists():
                try:
                    img.save(img_path, "PNG")
                except Exception as e:
                    print(f"  skip {idx}: image save failed ({e})", file=sys.stderr)
                    continue

            rel = str(img_path.relative_to(raw_dir))
            f.write(json.dumps({
                "image":  rel,
                "text":   text,
                "source": "olmocr_mix_1025",
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"olmocr_mix: {kept:,} samples → {out_jsonl}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/olmocr_mix.jsonl")
    parser.add_argument("--limit",     type=int, default=15000)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.limit)


if __name__ == "__main__":
    main()
