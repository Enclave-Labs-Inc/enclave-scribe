"""ai4bharat/indicdlp — page images for iter-4 hybrid bootstrap.

Iter-4 uses this dataset as the SOURCE of real page images. It ships with
119k human-annotated Indic document pages across 12 languages. We only
care about the Devanagari-script subset (Hindi + Marathi).

The dataset was built for LAYOUT parsing (COCO bounding boxes), not OCR.
There are no text ground truths here — those come from iter-3 in the next
step (see label_pages_with_iter3.py). This script just extracts the
images.

Access:
- MIT license, but GATED — you must accept terms on the HF page:
  https://huggingface.co/datasets/ai4bharat/indicdlp
- Once accepted, our existing HF write token can download.

Output:
- data/raw/indicdlp_pages/<lang>/<shard>/<idx>.png
- data/interim/indicdlp_pages.manifest.jsonl — one line per image:
    {"image": "<lang>/<shard>/<idx>.png", "language": "hi", "source": "indicdlp"}

Gate 0 support: --sample_only N pulls a diverse subset (~half Hindi,
half Marathi) to data/raw/indicdlp_pages_sample/ for eyeballing BEFORE
committing to a bulk download + label spend.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


DATASET_ID = "ai4bharat/indicdlp"
DEFAULT_LANGS = ("hi", "mr")
SHARD_SIZE = 500


def _stream_indicdlp(langs: tuple[str, ...]):
    """Yield (index, sample) for samples matching one of the target languages.

    IndicDLP's exact field name for language is unknown ahead of time (dataset
    is gated). We try the common candidates and fall back to a diagnostic
    print if none match — same auto-diagnose pattern we used for himalaya-ai.
    """
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    lang_keys = ("language", "lang", "lang_code", "language_code")

    first_diag = True
    matched = 0
    scanned = 0
    for i, sample in enumerate(ds):
        scanned += 1
        if first_diag:
            print(f"[diagnose] first sample keys: {list(sample.keys())}")
            first_diag = False

        # Find language field
        lang = None
        for k in lang_keys:
            if k in sample:
                lang = str(sample[k]).strip().lower()
                break
        if lang is None:
            # Try scanning all string values for a 2-letter code — last resort
            for k, v in sample.items():
                if isinstance(v, str) and len(v) == 2 and v.lower() in langs:
                    lang = v.lower()
                    break

        if lang and lang in langs:
            matched += 1
            yield i, lang, sample

        if scanned % 5000 == 0:
            print(f"[diagnose] scanned {scanned:,}, matched {matched:,} so far "
                  f"({100*matched/scanned:.1f}% hit rate)")


def _get_image(sample) -> "PIL.Image.Image | None":
    """Pull the PIL image out of an IndicDLP sample, tolerating field variants."""
    for k in ("image", "img", "png", "jpg"):
        img = sample.get(k)
        if img is not None and hasattr(img, "convert"):
            return img
    return None


def run(
    raw_dir: Path,
    manifest_jsonl: Path,
    langs: tuple[str, ...],
    max_samples: int,
    sample_only: int = 0,
    seed: int = 42,
) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # For --sample_only mode: alternate langs so the 20 samples cover both
    rng = random.Random(seed)

    print(f"Streaming {DATASET_ID} filtered to {langs} (target {max_samples:,}) ...")
    written = 0
    per_lang_count = {l: 0 for l in langs}
    shard = 0
    shard_dir = raw_dir / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    with open(manifest_jsonl, "w", encoding="utf-8") as mf:
        pbar = tqdm(total=max_samples, desc="pages saved")
        for _, lang, sample in _stream_indicdlp(langs):
            if written >= max_samples:
                break

            # In sample_only mode, keep languages balanced
            if sample_only:
                target = max_samples // len(langs)
                if per_lang_count[lang] >= target:
                    continue

            img = _get_image(sample)
            if img is None:
                continue
            try:
                img = img.convert("RGB")
            except Exception:
                continue

            if written > 0 and written % SHARD_SIZE == 0:
                shard += 1
                shard_dir = raw_dir / f"{shard:04d}"
                shard_dir.mkdir(exist_ok=True)

            fname = f"{written % SHARD_SIZE:05d}_{lang}.png"
            img_path = shard_dir / fname
            img.save(img_path, "PNG")

            rel = str(img_path.relative_to(raw_dir))
            mf.write(json.dumps({
                "image":    rel,
                "language": lang,
                "source":   "indicdlp",
            }) + "\n")

            per_lang_count[lang] += 1
            written += 1
            pbar.update(1)
        pbar.close()

    print(f"\nWrote {written:,} images total:")
    for l, n in per_lang_count.items():
        print(f"  {l}: {n:,}")
    print(f"Manifest → {manifest_jsonl}")
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",       default="data/raw/indicdlp_pages")
    p.add_argument("--manifest_jsonl", default="data/interim/indicdlp_pages.manifest.jsonl")
    p.add_argument("--langs",          nargs="+", default=list(DEFAULT_LANGS))
    p.add_argument("--max_samples",    type=int, default=3_500,
                   help="Cap on total images to download")
    p.add_argument("--sample_only",    type=int, default=0,
                   help="Gate 0 mode: pull only N samples (balanced across langs) for eyeball QA. Recommended: 20.")
    p.add_argument("--seed",           type=int, default=42)
    args = p.parse_args()

    if args.sample_only:
        # Override output paths and cap so sample and bulk modes don't collide
        args.raw_dir = str(Path(args.raw_dir).with_name(Path(args.raw_dir).name + "_sample"))
        args.manifest_jsonl = str(Path(args.manifest_jsonl).with_name(
            Path(args.manifest_jsonl).stem + "_sample.jsonl"))
        args.max_samples = args.sample_only
        print(f"Gate 0 mode: writing {args.sample_only} samples to {args.raw_dir}")

    run(
        raw_dir=Path(args.raw_dir),
        manifest_jsonl=Path(args.manifest_jsonl),
        langs=tuple(l.lower() for l in args.langs),
        max_samples=args.max_samples,
        sample_only=args.sample_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
