"""One-shot stager for the 500-sample Devanagari word benchmark.

Iter-7a declared the Devanagari-regression ship gate (`himalaya_500` word CER
≤ 28.1%) as *hard* but shipped without testing it, because the 500 images
referenced by `data/benchmark/iter3_words_500.jsonl` weren't on S3. Their
paths look like `himalaya_indic/0004/04800.jpg` — the underlying dataset is
`himalaya-ai/devanagari_ocr_dataset` (~58 GB tar.gz shards + 2 GB annotations
JSON), way too big to download every iter for 500 tiny word crops.

This script runs `prep_himalaya_indic.py` once with the exact `--max_samples`
needed to cover the 6 shards our 500 target images span (0000-0005 → ~30k
samples), then extracts *just those 500 images* into a small
`data/benchmark/himalaya_500/` directory and uploads that to S3. Every
future iter's Devanagari gate is then a $2 eval.

USAGE
    # On an AWS instance (60 GB scratch fits fine, ~30 min):
    python scripts/prepare/stage_devanagari_benchmark.py \\
        --gt_jsonl data/benchmark/iter3_words_500.jsonl \\
        --s3_bucket enclave-scribe-checkpoints \\
        --s3_prefix data/benchmark/himalaya_500

    # Locally, verify the plan without running prep or hitting S3:
    python scripts/prepare/stage_devanagari_benchmark.py --dry_run

FIXED IDEMPOTENCY
    - `prep_himalaya_indic.py` already skips per-image saves when the file
      exists on disk; re-running is cheap.
    - The final `aws s3 sync` only uploads what's changed.
    - The output `data/benchmark/himalaya_500.jsonl` mirrors the input
      `iter3_words_500.jsonl` but renamed to the ship-gate name.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GT = "data/benchmark/iter3_words_500.jsonl"
DEFAULT_S3 = "s3://enclave-scribe-checkpoints/data/benchmark/himalaya_500"


def _read_targets(gt_jsonl: Path) -> list[dict]:
    """Load the 500-sample benchmark JSONL — each record has {image, text, ...}."""
    if not gt_jsonl.exists():
        raise SystemExit(
            f"missing {gt_jsonl}. Sync it first:\n"
            f"  aws s3 cp s3://enclave-scribe-checkpoints/{gt_jsonl.as_posix().split('/', 1)[-1]} {gt_jsonl}"
        )
    records = []
    with open(gt_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _plan(records: list[dict]) -> tuple[list[str], int]:
    """From records, return (unique image paths, max_samples needed for prep_himalaya)."""
    paths = sorted({r["image"] for r in records})
    max_index_per_shard: dict[str, int] = {}
    for p in paths:
        # e.g. "himalaya_indic/0004/04800.jpg"
        parts = p.split("/")
        shard = parts[-2]
        idx = int(parts[-1].split(".")[0])
        max_index_per_shard[shard] = max(max_index_per_shard.get(shard, 0), idx)
    # prep_himalaya_indic writes 5000 samples per shard. To guarantee we cover
    # every referenced (shard, index), download through the highest shard fully.
    highest_shard = max(int(s) for s in max_index_per_shard)
    max_samples_needed = (highest_shard + 1) * 5000  # 6 shards → 30_000
    return paths, max_samples_needed


def _run_prep(raw_dir: Path, max_samples: int) -> None:
    """Run scripts/prepare/prep_himalaya_indic.py. Streams from HF WebDataset;
    ~7-8 GB per shard × 6 shards = ~45 GB + 2 GB annotations."""
    print(f"\n=== Running prep_himalaya_indic.py (max_samples={max_samples:,}) ===")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/prepare/prep_himalaya_indic.py",
            "--raw_dir", str(raw_dir),
            "--out_jsonl", "data/interim/himalaya_indic.jsonl",
            "--max_samples", str(max_samples),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"prep_himalaya_indic.py exited with code {result.returncode}")


def _verify_and_copy(raw_dir: Path, paths: list[str], out_dir: Path) -> tuple[int, list[str]]:
    """Copy each target image from raw_dir/<rel> to out_dir/<rel>.
    Returns (successful_count, missing_paths)."""
    missing: list[str] = []
    ok = 0
    for rel in paths:
        src = raw_dir / rel
        dst = out_dir / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        ok += 1
    return ok, missing


def _write_mirror_jsonl(records: list[dict], out_path: Path) -> None:
    """Mirror the benchmark JSONL under the ship-gate name (himalaya_500.jsonl)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _s3_sync(local_dir: Path, s3_uri: str) -> None:
    print(f"\n=== Uploading {local_dir} → {s3_uri} ===")
    result = subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_uri],
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"aws s3 sync exited with code {result.returncode}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--gt_jsonl", default=DEFAULT_GT,
                   help=f"Source benchmark JSONL (default: {DEFAULT_GT})")
    p.add_argument("--raw_dir", default="data/raw",
                   help="Where prep_himalaya_indic will write the full himalaya_indic tree")
    p.add_argument("--out_dir", default="data/benchmark/himalaya_500",
                   help="Where the 500-image subset is assembled locally")
    p.add_argument("--out_jsonl", default="data/benchmark/himalaya_500.jsonl",
                   help="Mirror of the input JSONL under the ship-gate name")
    p.add_argument("--s3_uri", default=DEFAULT_S3,
                   help=f"S3 destination for the 500-image subset (default: {DEFAULT_S3})")
    p.add_argument("--dry_run", action="store_true",
                   help="Plan only — read the JSONL, print what would be downloaded/copied/uploaded")
    p.add_argument("--skip_prep", action="store_true",
                   help="Assume raw_dir/himalaya_indic/ already has the images; go straight to copy+upload")
    args = p.parse_args()

    records = _read_targets(REPO_ROOT / args.gt_jsonl)
    paths, max_samples = _plan(records)
    print(f"Target images: {len(paths)} unique paths across "
          f"{len({p.split('/')[-2] for p in paths})} shards")
    print(f"prep_himalaya_indic --max_samples needed: {max_samples:,}")
    print(f"Estimated download: ~{(max_samples // 5000) * 8} GB (tar shards + 2 GB metadata JSON)")

    if args.dry_run:
        print("\n[dry-run] would run:")
        print(f"  1. python scripts/prepare/prep_himalaya_indic.py --max_samples {max_samples}")
        print(f"  2. copy {len(paths)} files from {args.raw_dir} → {args.out_dir}")
        print(f"  3. write {args.out_jsonl} ({len(records)} lines)")
        print(f"  4. aws s3 sync {args.out_dir} {args.s3_uri}")
        return

    raw_dir = REPO_ROOT / args.raw_dir
    out_dir = REPO_ROOT / args.out_dir

    if not args.skip_prep:
        _run_prep(raw_dir, max_samples)

    ok, missing = _verify_and_copy(raw_dir, paths, out_dir)
    print(f"\nCopied {ok}/{len(paths)} images to {out_dir}")
    if missing:
        print(f"WARN: {len(missing)} images missing from {raw_dir}; first few:", file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        if len(missing) > 10:
            raise SystemExit(f"too many missing images ({len(missing)}); check prep_himalaya_indic ran cleanly")

    _write_mirror_jsonl(records, REPO_ROOT / args.out_jsonl)
    print(f"Wrote mirrored JSONL: {args.out_jsonl}")

    _s3_sync(out_dir, args.s3_uri)
    # Also upload the mirrored JSONL to the benchmark bucket root so every
    # future iter's aws s3 sync data/benchmark/ picks it up.
    jsonl_uri = f"s3://enclave-scribe-checkpoints/{args.out_jsonl}"
    subprocess.run(["aws", "s3", "cp", str(REPO_ROOT / args.out_jsonl), jsonl_uri], check=True)
    print(f"Uploaded JSONL: {jsonl_uri}")

    print("\nDone. Every future iter can now Devanagari-eval with:")
    print(f"  aws s3 sync {args.s3_uri} data/raw/{args.out_dir.replace('data/benchmark/', 'himalaya_indic/')}")
    print(f"  # or point --image_root at wherever the sync lands")


if __name__ == "__main__":
    main()
