"""himalaya-ai/devanagari_ocr_dataset — page-level filter for iter-4.

Iter-3 used the WORD-level subset of this dataset (single Devanagari word
crops). Iter-4 needs PAGE-level Devanagari to teach the model long-context
document structure. The same dataset bundles page-level content from
several sources (per the HF dataset card, verified 2026-08-31):

  Page-level sources present in himalaya-ai:
    - krutrim-ai-labs/IndicVisionBench
    - Nayana-cognitivelab/NayanaBench
    - Malathip72/devanagari-ocr-dataset
    - gauravgiri/nepali-ocr-dataset

  Word-level sources (excluded here):
    - darknight054/indic-mozhi-ocr
    - c3rl/IIIT-INDIC-HW-WORDS-Hindi
    - rockerritesh/devanagari_and_roman_digits

The upstream schema is the same as iter-3 (see prep_himalaya_indic.py):
  - Streaming tar.gz shards yielding {__key__, __url__, png}
  - Separate ~2 GB devanagari_ocr.json with 7.5M entries keyed by image path

We can't tell page vs word from the __key__ alone. Two-phase approach:

  Phase 1 (auto-diagnose, runs on every invocation before filtering):
    - Scan the annotations JSON
    - Print top-20 stem prefixes with count + median text length
    - This tells us which prefixes correspond to page-level sources

  Phase 2 (filter + emit):
    - Include only stems matching --include_prefixes (default: known page-level
      prefixes) OR passing the min_text_chars threshold (default: 100 chars).
    - Anything shorter than min_text_chars is treated as word/short and dropped.

If the auto-diagnose output shows a different prefix pattern than expected,
adjust --include_prefixes accordingly.
"""
import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm


DATASET_ID     = "himalaya-ai/devanagari_ocr_dataset"
ANNOTATION_FN  = "devanagari_ocr.json"
SHARD_SIZE     = 500

# Best-guess page-level prefixes. Adjust after seeing the diagnose output on
# the first real run — the exact prefixes depend on how himalaya-ai named the
# images from each source dataset.
DEFAULT_PAGE_PREFIXES = (
    "indicvision",
    "krutrim",
    "nayana",
    "malathip",
    "devanagari_ocr",
    "nepali_ocr",
    "gauravgiri",
)

# Pages have real text, not single words. Anything under this is almost
# certainly a word crop that leaked past the prefix filter.
DEFAULT_MIN_TEXT_CHARS = 100

_PREFIX_SPLIT = re.compile(r"[_\-/]")


def _stem_prefix(stem: str, depth: int = 2) -> str:
    """Extract a stable prefix from an image stem.

    'indicvisionbench_hindi_page_00042' -> 'indicvisionbench_hindi'
    'nayana_layout_seal_0007'           -> 'nayana_layout'
    'nepali_local_1285138'              -> 'nepali_local'
    """
    parts = _PREFIX_SPLIT.split(stem)
    return "_".join(parts[:depth]) if len(parts) >= depth else stem


def _build_annotation_index(ann_path: Path) -> dict[str, str]:
    """Read the annotations JSON, return {image_stem: assistant_text}."""
    print(f"Loading {ann_path.name} ...")
    with open(ann_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    lookup: dict[str, str] = {}
    for e in tqdm(entries, desc="indexing"):
        msgs = e.get("messages") or []
        text = ""
        for m in msgs:
            if str(m.get("role", "")).lower() in ("assistant", "gpt"):
                text = str(m.get("content", "")).strip()
                break
        if not text:
            continue
        for img_ref in e.get("images") or []:
            stem = Path(str(img_ref)).stem
            if stem:
                lookup[stem] = text
    print(f"Indexed {len(lookup):,} image stems")
    return lookup


def _diagnose(lookup: dict[str, str], top_k: int = 20) -> None:
    """Auto-diagnose: print top-K stem prefixes with count + median text length.

    Use this output to decide which prefixes are page-level sources.
    """
    by_prefix: dict[str, list[int]] = defaultdict(list)
    for stem, text in lookup.items():
        by_prefix[_stem_prefix(stem, depth=2)].append(len(text))

    print(f"\n=== PREFIX DISTRIBUTION (top {top_k} by count) ===")
    print(f"{'prefix':<30s}  {'count':>10s}  {'median_chars':>13s}  {'p90_chars':>10s}")
    print("-" * 70)
    ranked = sorted(by_prefix.items(), key=lambda kv: -len(kv[1]))
    for prefix, lengths in ranked[:top_k]:
        n = len(lengths)
        med = int(statistics.median(lengths))
        p90 = int(sorted(lengths)[int(0.9 * n)]) if n > 1 else med
        print(f"{prefix:<30s}  {n:>10,d}  {med:>13d}  {p90:>10d}")
    print("=" * 70)


def _keep(stem: str, text: str, include_prefixes: tuple[str, ...],
          min_text_chars: int) -> bool:
    if len(text) < min_text_chars:
        return False
    if not include_prefixes:
        return True
    stem_lower = stem.lower()
    return any(stem_lower.startswith(p.lower()) for p in include_prefixes)


def run(
    raw_dir: Path,
    out_jsonl: Path,
    benchmark_jsonl: Path | None,
    include_prefixes: tuple[str, ...],
    min_text_chars: int,
    max_samples: int,
    val_ratio: float,
    seed: int,
) -> int:
    rng = random.Random(seed)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ANNOTATION_FN} (~2 GB, one-time HF cache) ...")
    ann_path = Path(hf_hub_download(DATASET_ID, ANNOTATION_FN, repo_type="dataset"))
    lookup = _build_annotation_index(ann_path)

    _diagnose(lookup)

    print(f"\nFilters:")
    print(f"  include_prefixes = {include_prefixes or '(none — text length only)'}")
    print(f"  min_text_chars   = {min_text_chars}")

    kept_lookup = {s: t for s, t in lookup.items()
                   if _keep(s, t, include_prefixes, min_text_chars)}
    print(f"  → {len(kept_lookup):,} of {len(lookup):,} entries pass the filter")

    if not kept_lookup:
        print("\nNo entries passed the filter. Check the diagnose output above")
        print("and rerun with --include_prefixes matching the real page sources.")
        return 0

    # Stream tar shards, save only matched samples
    print(f"\nStreaming {DATASET_ID} (target {max_samples:,} page samples) ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    img_base = raw_dir / "himalaya_pages"
    img_base.mkdir(parents=True, exist_ok=True)

    train_count = test_count = seen = unmatched = broken = shard = 0
    shard_dir = img_base / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    train_f = open(out_jsonl, "w", encoding="utf-8")
    test_f = open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else None

    try:
        pbar = tqdm(total=max_samples, desc="pages matched")
        for sample in ds:
            if train_count + test_count >= max_samples:
                break
            seen += 1

            key = str(sample.get("__key__", "")).strip()
            text = kept_lookup.get(key)
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
                "prompt": "Extract this document page as clean markdown. "
                          "Preserve the original script exactly.",
            }
            if is_test:
                record["category"] = "himalaya_pages"

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

    print(f"\nhimalaya pages: {train_count:,} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"himalaya pages: {test_count:,} held out → {benchmark_jsonl}")
    print(f"seen: {seen:,}  unmatched: {unmatched:,}  broken: {broken:,}")
    return train_count + test_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",         default="data/raw")
    p.add_argument("--out_jsonl",       default="data/interim/himalaya_pages.jsonl")
    p.add_argument("--benchmark_jsonl", default="",
                   help="If set, val_ratio of samples routed here")
    p.add_argument("--include_prefixes", nargs="*", default=list(DEFAULT_PAGE_PREFIXES),
                   help="Only keep stems starting with these (case-insensitive). "
                        "Pass an empty string to disable prefix filtering.")
    p.add_argument("--min_text_chars",  type=int, default=DEFAULT_MIN_TEXT_CHARS)
    p.add_argument("--max_samples",     type=int, default=3_000)
    p.add_argument("--val_ratio",       type=float, default=0.05)
    p.add_argument("--seed",            type=int, default=42)
    args = p.parse_args()

    prefixes = tuple(p for p in args.include_prefixes if p) or ()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(
        raw_dir=Path(args.raw_dir),
        out_jsonl=Path(args.out_jsonl),
        benchmark_jsonl=bench,
        include_prefixes=prefixes,
        min_text_chars=args.min_text_chars,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
