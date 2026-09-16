"""Assemble the iter-10 bilingual (English + Indic) training corpus (~40k samples).

Iter-10 attacks the plateau iter-9 confirmed: OCRBench V2 has 4 task families
(text spotting, text referring, relation extraction, element parsing across 31
scenarios; arXiv 2501.00321). Iter-7a's DocVQA-heavy corpus trained one slice;
iter-9's 87%-DocVQA re-run confirmed adding more of the same is null.

MIX (task-shape matched to OCRBench V2)
    - olmocr_mix              15,000  (allenai/olmOCR-mix-1025 subset, full-page English)
    - pubtabnet                5,000  (table structure — element parsing)
    - chartqa                  5,000  (chart parsing — element parsing)
    - math_formula             3,000  (LaTeX-OCR — element parsing)
    - docvqa                   8,000  (short-answer VQA — retain iter-7a's strength)
    - xfund                    1,400  (multilingual forms)
    - devanagari_synthetic     2,000  (full-page synthetic Hindi via SynthTIGER + Noto/Mangal)
    - himalaya_replay            500  (word-level Devanagari, iter-4 anti-forgetting)
    Total: ~40,000 samples. English:Indic ≈ 85:15.

FLOW
    1. Run each prep script (skippable with --skip_prep if jsonls already exist).
       Each writes to data/interim/<name>.jsonl.
    2. Load, subsample per source, tag with `source` field.
    3. Dedup by image path, filter empty text, drop len > 3000 chars.
    4. Shuffle (seed=42), split 98/2 train/val.
    5. Write data/processed/iter10_train.jsonl + iter10_val.jsonl.

DATA HYGIENE (carried forward from iter-7a/iter-9 lessons)
    - S3 mirror ALL sources first (via mirror_datasets_to_s3.DatasetMirror).
      HF rate limits burned $10-16 per iter in iter-7a and iter-9.
    - Cap every prep at 15k default per source (iter-9's IDL 100k default
      burned ~$10 on samples that got dropped).
    - Drop text > 3000 chars (iter-7a OOM lesson).

USAGE
    # Full run (kicks every prep; assumes S3 mirrors exist — see mirror_datasets_to_s3.py)
    python scripts/prepare/build_iter10_corpus.py

    # Skip prep (assumes data/interim/*.jsonl already present)
    python scripts/prepare/build_iter10_corpus.py --skip_prep

    # Dry-run — validate config and per-source counts, no writes
    python scripts/prepare/build_iter10_corpus.py --dry_run
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
# None script = static / already-produced source (himalaya_replay from data/benchmark/)
SOURCES: list[tuple[str, str | None, str, int]] = [
    ("olmocr_mix",           "scripts/prepare/prep_olmocr_mix.py",       "data/interim/olmocr_mix.jsonl",             15_000),
    ("pubtabnet",            "scripts/prepare/prep_pubtabnet.py",        "data/interim/pubtabnet.jsonl",               5_000),
    ("chartqa",              "scripts/prepare/prep_chartqa.py",          "data/interim/chartqa.jsonl",                 5_000),
    ("math_formula",         "scripts/prepare/prep_math_formula.py",     "data/interim/math_formula.jsonl",            3_000),
    ("docvqa",               "scripts/prepare/prep_docvqa.py",           "data/interim/docvqa.jsonl",                  8_000),
    ("xfund",                "scripts/prepare/prep_xfund.py",            "data/interim/xfund.jsonl",                   1_400),
    ("devanagari_synthetic", "scripts/prepare/synth_devanagari_pages.py","data/interim/devanagari_synthetic.jsonl",   2_000),
    ("himalaya_replay",      None,                                        "data/benchmark/himalaya_500.jsonl",            500),
]


MAX_TEXT_LEN = 3000  # iter-7a OOM lesson — drop samples above this


def _run_prep(script: str, interim: Path) -> None:
    """Run a prep script from repo root; skip if interim already populated.

    Crash-recovery-friendly: a partial prep failure won't force re-download
    of everything (each interim is checkpointed).
    """
    if interim.exists() and interim.stat().st_size > 0:
        print(f"\n=== SKIP {script}: {interim} already populated "
              f"({interim.stat().st_size} bytes) ===", flush=True)
        return
    print(f"\n=== Running {script} ===", flush=True)
    result = subprocess.run(
        [sys.executable, script],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        # Don't hard-abort — a single missing source is not fatal (drop-and-warn pattern)
        print(f"WARN: {script} exited with code {result.returncode} — will proceed without it",
              file=sys.stderr, flush=True)


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
        r = dict(r)
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


def _drop_empty_and_long(records: list[dict]) -> list[dict]:
    out = []
    for r in records:
        text = r.get("text", "")
        if not text or not text.strip():
            continue
        if len(text) > MAX_TEXT_LEN:
            continue
        out.append(r)
    return out


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

    per_source_counts: dict[str, int] = {}
    all_samples: list[dict] = []
    for name, _, jsonl, cap in SOURCES:
        path = REPO_ROOT / jsonl
        records = _load_jsonl(path)
        if not records:
            print(f"WARN: {jsonl} empty or missing — dropping {name} from corpus",
                  file=sys.stderr, flush=True)
            per_source_counts[name] = 0
            continue
        capped = _subsample(records, cap, rng)
        tagged = _tag(capped, name)
        all_samples.extend(tagged)
        per_source_counts[name] = len(tagged)
        print(f"  {name:<24} {len(records):>8,} available → {len(tagged):>8,} taken")

    print(f"\nRaw total: {len(all_samples):,} samples")

    all_samples = _dedup_by_image(all_samples)
    print(f"After dedup by image path: {len(all_samples):,}")
    all_samples = _drop_empty_and_long(all_samples)
    print(f"After drop empty + len > {MAX_TEXT_LEN}: {len(all_samples):,}")

    rng.shuffle(all_samples)

    n_val = max(200, int(len(all_samples) * val_ratio))
    val = all_samples[:n_val]
    train = all_samples[n_val:]

    if dry_run:
        print("\n[dry-run] no files written. Summary:")
        for name, count in per_source_counts.items():
            print(f"  {name:<24} {count:>8,}")
        print(f"  train                    {len(train):>8,}")
        print(f"  val                      {len(val):>8,}")
        return

    train_out.parent.mkdir(parents=True, exist_ok=True)
    for path, samples in [(train_out, train), (val_out, val)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in samples:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nWrote:")
    print(f"  train : {len(train):>8,} → {train_out}")
    print(f"  val   : {len(val):>8,} → {val_out}")
    print("\nPer-source (before dedup+filter):")
    for name, count in per_source_counts.items():
        print(f"  {name:<24} {count:>8,}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train_out", default="data/processed/iter10_train.jsonl")
    p.add_argument("--val_out",   default="data/processed/iter10_val.jsonl")
    p.add_argument("--val_ratio", type=float, default=0.02)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--skip_prep", action="store_true",
                   help="Skip running prep scripts; assume interim jsonls exist")
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
