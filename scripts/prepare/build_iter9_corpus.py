"""Assemble the iter-9 VQA-heavy training corpus (~120k samples).

Iter-9 = Path A (VQA specialization) per plan 2026-09-14. Extends iter-7a's
accidental OCRBench-V2 win (F1 0.499, 70.6% of Interfaze) by *adding* real
VQA / OCR-VQA training data on top of iter-7a's DocVQA + XFUND + IDL mix.

MIX
    - iter7a-baseline   ~10k  (docvqa + xfund + idl + himalaya replay — reruns iter-7a preps)
    - ocrvqa            ~40k  (howard-hou/OCR-VQA)
    - chartqa           ~25k  (ahmed-masry/ChartQA)
    - textvqa           ~30k  (lmms-lab/textvqa — audit swap from `textvqa/textvqa` which is 401)
    - infographicvqa    ~5k   (HuggingFaceM4/InfographicVQA — audit swap; drops to 0 if load fails)
    - docvqa-extra      ~10k  (additional unseen DocVQA samples on top of iter-7a's set)
    - himalaya-replay   ~500  (Devanagari word replay — anti-forgetting)
    ~120k target; 12× iter-7a's post-filter 10k corpus.

FORMAT
    Every non-DocVQA sample stores its per-task question in `sample["prompt"]`
    and answer text in `sample["text"]`. Iter-8's eval-side fix + iter-7a's
    collator.py:18 `item.get("prompt", self.prompt)` both route this correctly.

APPROACH
    Warm-start iter-9 from iter-7a adapter with LR 5e-5, 1 epoch. This is
    EXTENSION not full retrain — keep iter-7a's short-answer VQA behavior,
    add more VQA-shaped data.

FLOW
    1. Run each prep script (skippable via --skip_prep when interim jsonls exist).
       Each writes to data/interim/<name>.jsonl.
    2. Load, subsample per source, tag with `source` field.
    3. Dedup by image path, filter empty text.
    4. Shuffle, split 98/2 train/val.
    5. Write data/processed/iter9_train.jsonl + iter9_val.jsonl.

USAGE
    # Full run — kicks every prep (hours of downloads)
    python scripts/prepare/build_iter9_corpus.py

    # Skip prep (assumes data/interim/*.jsonl already present)
    python scripts/prepare/build_iter9_corpus.py --skip_prep

    # Dry-run — validate config + per-source line counts, no writes
    python scripts/prepare/build_iter9_corpus.py --dry_run

DEVIATIONS FROM PLAN (2026-09-14 readiness audit)
    - `textvqa/textvqa` (401) → `lmms-lab/textvqa`
    - `vqa-infographic/infographicvqa` (401) → `HuggingFaceM4/InfographicVQA`;
      if THAT 401s at runtime, InfographicVQA is silently dropped (5k / 4%).
    - DocVQA-extra source disabled at load-time if the iter-7a docvqa.jsonl
      already contains >=40k samples (no unseen slice available).
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

# (source name, prep script relative to repo root or None, interim jsonl, target cap, extra args)
SOURCES: list[tuple[str, str | None, str, int, list[str]]] = [
    ("docvqa",         "scripts/prepare/prep_docvqa.py",         "data/interim/docvqa.jsonl",           10_000, []),
    ("xfund",          "scripts/prepare/prep_xfund.py",          "data/interim/xfund.jsonl",             5_000, []),
    ("idl",            "scripts/prepare/prep_idl.py",            "data/interim/idl.jsonl",              10_000, []),
    ("ocrvqa",         "scripts/prepare/prep_ocrvqa.py",         "data/interim/ocrvqa.jsonl",           40_000, ["--max_samples", "45000"]),
    ("chartqa",        "scripts/prepare/prep_chartqa.py",        "data/interim/chartqa.jsonl",          25_000, ["--max_samples", "28000"]),
    ("textvqa",        "scripts/prepare/prep_textvqa.py",        "data/interim/textvqa.jsonl",          30_000, ["--max_samples", "34000"]),
    ("infographicvqa", "scripts/prepare/prep_infographicvqa.py", "data/interim/infographicvqa.jsonl",    5_000, ["--max_samples", "6000"]),
    ("replay",         None,                                     "data/interim/himalaya_indic.jsonl",     500, []),
]


def _run_prep(script: str, interim: Path, extra_args: list[str]) -> None:
    """Run a prep script from the repo root; propagate its exit code.

    Skip when the interim JSONL already has content (idempotent re-runs after a
    partial failure — don't re-download OCR-VQA's 40k images just because
    ChartQA 404'd afterwards).
    """
    if interim.exists() and interim.stat().st_size > 0:
        print(f"\n=== SKIP {script}: {interim} already populated ({interim.stat().st_size} bytes) ===", flush=True)
        return
    print(f"\n=== Running {script} {' '.join(extra_args)} ===", flush=True)
    result = subprocess.run(
        [sys.executable, script, *extra_args],
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


def _dedup_by_image_and_prompt(records: list[dict]) -> list[dict]:
    """Dedup by (image, prompt). Same image with different questions = kept."""
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for r in records:
        key = (r.get("image", ""), r.get("prompt", ""))
        if not key[0] or key in seen:
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
        for _, script, jsonl, _, extra_args in SOURCES:
            if script is None:
                continue
            _run_prep(script, REPO_ROOT / jsonl, extra_args)

    # Load + subsample per source
    per_source_counts: dict[str, int] = {}
    all_samples: list[dict] = []
    for name, _, jsonl, cap, _ in SOURCES:
        path = REPO_ROOT / jsonl
        records = _load_jsonl(path)
        if name == "replay" and not records:
            print(f"WARN: {jsonl} missing — proceeding WITHOUT Devanagari replay "
                  "(500/120k = 0.4%, not load-bearing).",
                  file=sys.stderr, flush=True)
            per_source_counts[name] = 0
            continue
        if name == "infographicvqa" and not records:
            print(f"WARN: {jsonl} empty — InfographicVQA source dropped (per audit fallback). "
                  "5k/120k = 4% of corpus; remaining sources absorb the deficit.",
                  file=sys.stderr, flush=True)
            per_source_counts[name] = 0
            continue
        capped = _subsample(records, cap, rng)
        tagged = _tag(capped, name)
        all_samples.extend(tagged)
        per_source_counts[name] = len(tagged)
        print(f"  {name:<16} {len(records):>8,} available → {len(tagged):>8,} taken")

    print(f"\nRaw total: {len(all_samples):,} samples")

    # Dedup + drop empty
    all_samples = _dedup_by_image_and_prompt(all_samples)
    print(f"After dedup by (image, prompt): {len(all_samples):,}")
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
            print(f"  {name:<16} {count:>8,}")
        print(f"  train            {len(train):>8,}")
        print(f"  val              {len(val):>8,}")
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
        print(f"  {name:<16} {count:>8,} taken (before dedup/filter)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train_out", default="data/processed/iter9_train.jsonl")
    p.add_argument("--val_out",   default="data/processed/iter9_val.jsonl")
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
