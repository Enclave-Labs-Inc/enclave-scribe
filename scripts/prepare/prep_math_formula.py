"""Math formula images with LaTeX ground truth.

Target OCRBench V2's element-parsing / formula slice.

DATASET
    Primary : linxy/LaTeX_OCR   (HF-hosted, rendered LaTeX images + source)
    Fallback: unsloth/LaTeX_OCR, deepvk/LaTeX-OCR

OUTPUT
    data/raw/math_formula/<idx>.png
    data/interim/math_formula.jsonl

PROMPT
    "Extract the formula as LaTeX."

USAGE
    python scripts/prepare/prep_math_formula.py --limit 3000
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


PRIMARY_ID   = "linxy/LaTeX_OCR"
FALLBACK_IDS = ["unsloth/LaTeX_OCR", "deepvk/LaTeX-OCR"]

PROMPT = "Extract the formula as LaTeX."


def _try_load(dataset_id: str, split: str = "train"):
    from datasets import load_dataset
    try:
        return load_dataset(dataset_id, split=split, streaming=True)
    except Exception as e:
        print(f"  cannot load {dataset_id}: {e}", file=sys.stderr)
        return None


def run(raw_dir: Path, out_jsonl: Path, limit: int) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    img_dir = raw_dir / "math_formula"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = _try_load(PRIMARY_ID)
    if ds is None:
        for fb in FALLBACK_IDS:
            print(f"  falling back to {fb}", file=sys.stderr)
            ds = _try_load(fb)
            if ds is not None:
                break
    if ds is None:
        print("ERROR: no LaTeX-OCR source reachable; skipping", file=sys.stderr)
        return 0

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(ds, desc="math_formula", total=limit)):
            if kept >= limit:
                break
            latex = (
                sample.get("text") or sample.get("latex") or
                sample.get("formula") or sample.get("caption") or ""
            )
            latex = str(latex).strip()
            if not latex:
                continue

            img = sample.get("image") or sample.get("img")
            if img is None:
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
                "text":   latex,
                "prompt": PROMPT,
                "source": "math_formula",
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"math_formula: {kept:,} samples → {out_jsonl}")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/math_formula.jsonl")
    parser.add_argument("--limit",     type=int, default=3000)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.limit)


if __name__ == "__main__":
    main()
