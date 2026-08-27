"""himalaya-ai/devanagari_ocr_dataset — real Indic OCR data curated from 7 sources.

Dataset : himalaya-ai/devanagari_ocr_dataset (~58 GB tar.gz shards + 2 GB JSON)
Content : Hindi, Marathi, Sanskrit, Nepali, Pali, English word/page/document OCR
Sources : IIT-Indic-HW-Words, IndicVisionBench, NayanaBench, and 4 more
Output  : data/raw/himalaya_indic/<shard>/<idx>.jpg
          data/interim/himalaya_indic.jsonl
          data/benchmark/himalaya_indic_test.jsonl  (2% held out)

The upstream schema (discovered 2026-08-27 by probing a real sample):

  Streaming iterator yields WebDataset-style samples per tar.gz shard:
    {
      "__key__": "hindi_local_1285138",   # stem of the image filename
      "__url__": "hf://...images_batch_001.tar.gz",
      "png":     <PIL Image>,             # the cropped OCR image
    }

  Text annotations live in a SEPARATE devanagari_ocr.json file (2.15 GB,
  7.5M entries) keyed by image path in "messages" format:
    {
      "messages": [
        {"role": "user",      "content": "<image>Transcribe the Devanagari text..."},
        {"role": "assistant", "content": "स्वर्ग आफै"}
      ],
      "images":   ["devanagari_ocr/nepali_0.png"]
    }

We download the JSON once, build a lookup {image_stem → assistant_text}, then
iterate the tar.gz stream and match each sample by __key__.
"""
import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm


DATASET_ID     = "himalaya-ai/devanagari_ocr_dataset"
ANNOTATION_FN  = "devanagari_ocr.json"
SHARD_SIZE     = 5000
MIN_TEXT_CHARS = 1


def _build_annotation_lookup() -> dict[str, str]:
    """Download the annotations JSON and build {image_stem: text} lookup.

    Image paths in the JSON look like 'devanagari_ocr/nepali_0.png'.
    The tar.gz stream yields __key__ = 'nepali_0' (stem only).
    We key the lookup by stem so matching is trivial.
    """
    print(f"Downloading {ANNOTATION_FN} (first time only, ~2 GB) ...")
    ann_path = hf_hub_download(DATASET_ID, ANNOTATION_FN, repo_type="dataset")
    print(f"  cached at {ann_path}")
    print("Loading + building lookup dict (this can take a couple minutes) ...")

    with open(ann_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    lookup: dict[str, str] = {}
    for e in tqdm(entries, desc="indexing annotations"):
        # Extract assistant text from messages
        msgs = e.get("messages") or []
        assistant_text = ""
        for m in msgs:
            if str(m.get("role", "")).lower() in ("assistant", "gpt"):
                assistant_text = str(m.get("content", "")).strip()
                break
        if not assistant_text or len(assistant_text) < MIN_TEXT_CHARS:
            continue

        # Extract image stem from paths
        images = e.get("images") or []
        for img_ref in images:
            stem = Path(str(img_ref)).stem
            if stem:
                # Later entries win — safe because assistant_text is deterministic
                # per stem in this dataset.
                lookup[stem] = assistant_text

    print(f"Built lookup: {len(lookup):,} image stems -> text")
    return lookup


def run(
    raw_dir: Path,
    out_jsonl: Path,
    benchmark_jsonl: Path | None,
    max_samples: int = 30_000,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> int:
    rng = random.Random(seed)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: annotations lookup
    lookup = _build_annotation_lookup()

    # Step 2: tar.gz stream
    print(f"Streaming {DATASET_ID} (target {max_samples:,} matched samples) ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    img_base = raw_dir / "himalaya_indic"
    img_base.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    seen = 0
    unmatched = 0
    broken = 0
    shard = 0
    shard_dir = img_base / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    train_f = open(out_jsonl, "w", encoding="utf-8")
    test_f = open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else None

    try:
        pbar = tqdm(total=max_samples, desc="himalaya-ai matched")
        for sample in ds:
            if train_count + test_count >= max_samples:
                break
            seen += 1

            key = str(sample.get("__key__", "")).strip()
            if not key:
                unmatched += 1
                continue

            text = lookup.get(key)
            if not text:
                unmatched += 1
                continue

            img = sample.get("png")
            if img is None or not hasattr(img, "convert"):
                broken += 1
                continue
            try:
                img = img.convert("RGB")
            except Exception:
                broken += 1
                continue

            # Route to train or held-out
            is_test = (test_f is not None and rng.random() < val_ratio)
            target_f = test_f if is_test else train_f

            total = train_count + test_count
            if total > 0 and total % SHARD_SIZE == 0:
                shard += 1
                shard_dir = img_base / f"{shard:04d}"
                shard_dir.mkdir(exist_ok=True)

            fname = f"{total % SHARD_SIZE:05d}.jpg"
            img_path = shard_dir / fname
            img.save(img_path, "JPEG", quality=90)

            rel = str(img_path.relative_to(raw_dir))
            record = {
                "image":  rel,
                "text":   text,
                "prompt": "Transcribe the Devanagari text from this image:",
            }
            if is_test:
                record["category"] = "himalaya_indic"

            target_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if is_test:
                test_count += 1
            else:
                train_count += 1
                pbar.update(1)

        pbar.close()
    finally:
        train_f.close()
        if test_f is not None:
            test_f.close()

    print(f"\nhimalaya-ai: {train_count:,} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"himalaya-ai: {test_count:,} held out → {benchmark_jsonl}")
    print(f"seen: {seen:,}  unmatched: {unmatched:,}  broken: {broken:,}")
    return train_count + test_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--out_jsonl",       default="data/interim/himalaya_indic.jsonl")
    parser.add_argument("--benchmark_jsonl", default="",
                        help="If set, val_ratio of samples are routed here")
    parser.add_argument("--max_samples",     type=int, default=30_000)
    parser.add_argument("--val_ratio",       type=float, default=0.02)
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(
        Path(args.raw_dir),
        Path(args.out_jsonl),
        bench,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
