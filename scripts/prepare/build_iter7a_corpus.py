"""Assemble the iter-7a multilingual training corpus (~107k samples).

Iter-7a corrects the English-recall regression iter-6 diagnosed in iter-4
(OmniDocBench F1: base 0.291 vs iter-4 0.106) by tripling iter-2's corpus
with real English + multilingual OCR data.

MIX
    - docvqa    ~40k  (HuggingFaceM4/DocumentVQA, dense English docs)
    - textocr   ~25k  (natural-image dense text)
    - hiertext  ~12k  (hierarchical text structure)
    - xfund     ~10k  (7 langs: zh/ja/es/fr/it/de/pt)
    - idl       ~20k  (sampled from 250k historical documents)
    - replay    ~500  (Devanagari word crops, anti-forgetting)

FLOW
    1. Run each prep script (skippable with --skip_prep if jsonls already exist).
       Each writes to data/interim/<name>.jsonl.
    2. Load, subsample per source, tag with `source` field.
    3. Dedup by image path, filter empty text.
    4. Shuffle, split 98/2 train/val.
    5. Write data/processed/iter7a_train.jsonl + iter7a_val.jsonl.

REPLAY SOURCE
    prep_himalaya_indic.py needs a ~60 GB download for 500 samples, so if
    data/interim/himalaya_indic.jsonl is missing this script logs a
    warning and continues WITHOUT replay. 500/107k is 0.5% — the corpus
    won't miss it. The bootstrap script staged alongside this fetches the
    replay from S3 if available before invoking this builder.

USAGE
    # Full run (kicks every prep, hours of downloads)
    python scripts/prepare/build_iter7a_corpus.py

    # Skip prep (assumes data/interim/*.jsonl already present)
    python scripts/prepare/build_iter7a_corpus.py --skip_prep

    # Dry-run — validate config and per-source line counts, no writes
    python scripts/prepare/build_iter7a_corpus.py --dry_run
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

# (source name, prep script relative to repo root, interim jsonl, target sample cap)
SOURCES: list[tuple[str, str | None, str, int]] = [
    ("docvqa",   "scripts/prepare/prep_docvqa.py",   "data/interim/docvqa.jsonl",         40_000),
    # ("textocr", ...) DROPPED 2026-09-12: Open Images bucket returns 404 on per-image URLs
    #                   (dl.fbaipublicfiles.com JSON downloads fine, but image fetches fail silently
    #                   and the prep emits 0 samples with no error). Reinstate when TextOCR images
    #                   are relocated to a working host, or vendor them.
    # ("hiertext", ...) DROPPED 2026-09-12: https://storage.googleapis.com/hiertext/hiertext/*.jsonl.gz
    #                   returns 404 (Google moved the bucket). Reinstate when we point at gs://gresearch/hiertext/.
    ("xfund",    "scripts/prepare/prep_xfund.py",    "data/interim/xfund.jsonl",          10_000),
    # IDL bumped 20k → 40k on 2026-09-12 to partially compensate for dropped textocr (25k) + hiertext (12k)
    ("idl",      "scripts/prepare/prep_idl.py",      "data/interim/idl.jsonl",            40_000),
    ("replay",   None,                                "data/interim/himalaya_indic.jsonl",   500),
]


def _run_prep(script: str, interim: Path) -> None:
    """Run a prep script from the repo root; propagate its exit code.

    Skip when the interim JSONL already has content (idempotent re-runs after a
    partial failure — don't re-download DocVQA's 40k images just because HierText
    404'd afterwards).
    """
    if interim.exists() and interim.stat().st_size > 0:
        print(f"\n=== SKIP {script}: {interim} already populated ({interim.stat().st_size} bytes) ===", flush=True)
        return
    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"prep script {script} exited with code {result.returncode}")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _subsample(records: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if cap <= 0 or len(records) <= cap:
        return records
    rng.shuffle(records)
    return records[:cap]


def _tag(records: Iterable[dict], source: str) -> list[dict]:
    tagged = []
    for r in records:
        r = dict(r)  # shallow copy — don't mutate caller's dicts
        r["source"] = source
        tagged.append(r)
    return tagged


def _dedup_by_image(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    kept: list[dict] = []
    for r in records:
        key = r.get("image", "")
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(r)
    return kept


def _drop_empty(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("text", "").strip()]


def run(
    train_out: Path,
    val_out: Path,
    val_ratio: float,
    seed: int,
    skip_prep: bool,
    dry_run: bool,
) -> None:
    rng = random.Random(seed)

    if not skip_prep and not dry_run:
        for _, script, jsonl, _ in SOURCES:
            if script is None:
                continue
            _run_prep(script, REPO_ROOT / jsonl)

    # Load + subsample per source
    per_source_counts: dict[str, int] = {}
    all_samples: list[dict] = []
    for name, _, jsonl, cap in SOURCES:
        path = REPO_ROOT / jsonl
        records = _load_jsonl(path)
        if name == "replay" and not records:
            print(f"WARN: {jsonl} missing — proceeding WITHOUT Devanagari replay "
                  "(500/107k = 0.5%, not load-bearing). Fetch it via bootstrap if you need it.",
                  file=sys.stderr, flush=True)
            per_source_counts[name] = 0
            continue
        capped = _subsample(records, cap, rng)
        tagged = _tag(capped, name)
        all_samples.extend(tagged)
        per_source_counts[name] = len(tagged)
        print(f"  {name:<10} {len(records):>8,} available → {len(tagged):>8,} taken")

    print(f"\nRaw total: {len(all_samples):,} samples")

    # Dedup + drop empty
    all_samples = _dedup_by_image(all_samples)
    print(f"After dedup by image path: {len(all_samples):,}")
    all_samples = _drop_empty(all_samples)
    print(f"After dropping empty text: {len(all_samples):,}")

    rng.shuffle(all_samples)

    # Split
    n_val = max(1, int(len(all_samples) * val_ratio))
    val = all_samples[:n_val]
    train = all_samples[n_val:]

    if dry_run:
        print("\n[dry-run] no files written. Summary:")
        for name, count in per_source_counts.items():
            print(f"  {name:<10} {count:>8,}")
        print(f"  train      {len(train):>8,}")
        print(f"  val        {len(val):>8,}")
        return

    train_out.parent.mkdir(parents=True, exist_ok=True)
    for path, samples in [(train_out, train), (val_out, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in samples:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nWrote:")
    print(f"  train : {len(train):>8,} → {train_out}")
    print(f"  val   : {len(val):>8,} → {val_out}")
    print("\nPer-source (post-dedup+filter effect distributed):")
    for name, count in per_source_counts.items():
        print(f"  {name:<10} {count:>8,} taken (before dedup/filter)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train_out", default="data/processed/iter7a_train.jsonl")
    p.add_argument("--val_out",   default="data/processed/iter7a_val.jsonl")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--skip_prep", action="store_true",
                   help="Skip running prep scripts; assume data/interim/*.jsonl already exist")
    p.add_argument("--dry_run",   action="store_true",
                   help="Load interim jsonls, print counts, DO NOT write output or run preps")
    args = p.parse_args()

    run(
        train_out=Path(args.train_out),
        val_out=Path(args.val_out),
        val_ratio=args.val_ratio,
        seed=args.seed,
        skip_prep=args.skip_prep,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
