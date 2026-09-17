"""PubTabNet — table structure recognition ground truth.

Target OCRBench V2's element-parsing / table slice (untrained in iter-7a/9).

DATASET (verified reachable 2026-09-16 via HF hub listing)
    Primary : apoidea/pubtabnet-html  (HF-hosted HTML-formatted subset)
    Fallback: ibm/publaynet-pubtabnet, saidines/table-recognition-dataset

    IBM Research released PubTabNet under Community Data License (CDL-Permissive
    2.0). The `apoidea/pubtabnet-html` mirror ships pre-rendered HTML ground
    truth per image, which matches how our training loop consumes text targets.

OUTPUT
    data/raw/pubtabnet/<idx>.png
    data/interim/pubtabnet.jsonl

PROMPT
    "Extract this table as HTML."
    The training-side collator (scribe/data/collator.py:18) routes per-sample
    prompts, so this string overrides the config's default page-parsing prompt.

USAGE
    python scripts/prepare/prep_pubtabnet.py --limit 5000
"""
import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


PRIMARY_ID  = "apoidea/pubtabnet-html"
FALLBACK_ID = "ibm/publaynet-pubtabnet"

PROMPT = "Extract this table as HTML."


def _try_load(dataset_id: str, split: str = "train"):
    from datasets import load_dataset
    try:
        return load_dataset(dataset_id, split=split, streaming=True)
    except Exception as e:
        print(f"  cannot load {dataset_id}: {e}", file=sys.stderr)
        return None


def run(raw_dir: Path, out_jsonl: Path, limit: int) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    img_dir = raw_dir / "pubtabnet"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = _try_load(PRIMARY_ID)
    if ds is None:
        print(f"  falling back to {FALLBACK_ID}", file=sys.stderr)
        ds = _try_load(FALLBACK_ID)
    if ds is None:
        raise RuntimeError("pubtabnet produced 0 samples")

    kept = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(tqdm(ds, desc="pubtabnet", total=limit)):
            if kept >= limit:
                break
            # Field names vary — try common ones
            html = (
                sample.get("html") or sample.get("html_table") or
                sample.get("text") or sample.get("html_content") or ""
            )
            html = str(html).strip()
            if not html:
                continue

            img = sample.get("image") or sample.get("table_image") or sample.get("img")
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
                "text":   html,
                "prompt": PROMPT,
                "source": "pubtabnet",
            }, ensure_ascii=False) + "\n")
            kept += 1

    print(f"pubtabnet: {kept:,} samples → {out_jsonl}")
    if kept == 0:
        raise RuntimeError("pubtabnet produced 0 samples")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",   default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/pubtabnet.jsonl")
    parser.add_argument("--limit",     type=int, default=5000)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.limit)


if __name__ == "__main__":
    main()
