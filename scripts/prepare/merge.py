"""Merge all per-dataset JSONL files into train.jsonl + val.jsonl.

Usage:
    python scripts/data/merge.py \
        --interim_dir data/interim \
        --out_dir data/processed \
        --val_ratio 0.02
"""
import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def run(interim_dir: Path, out_dir: Path, val_ratio: float, seed: int = 42) -> None:
    jsonl_files = sorted(interim_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {interim_dir}")

    print("Loading datasets:")
    all_samples = []
    for path in jsonl_files:
        before = len(all_samples)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_samples.append(json.loads(line))
        added = len(all_samples) - before
        print(f"  {path.stem:<20} {added:>8,} samples")

    print(f"\nTotal: {len(all_samples):,} samples")

    # Deduplicate by image path
    seen = set()
    unique = []
    for s in all_samples:
        key = s["image"]
        if key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"After dedup: {len(unique):,} samples")

    # Filter empty text
    unique = [s for s in unique if s.get("text", "").strip()]
    print(f"After filtering empty text: {len(unique):,} samples")

    # Shuffle + split
    random.seed(seed)
    random.shuffle(unique)

    n_val = max(1, int(len(unique) * val_ratio))
    val_samples = unique[:n_val]
    train_samples = unique[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    for path, samples in [(train_path, train_samples), (val_path, val_samples)]:
        with open(path, "w", encoding="utf-8") as f:
            for s in tqdm(samples, desc=f"Writing {path.name}"):
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nDone:")
    print(f"  Train : {len(train_samples):>8,} → {train_path}")
    print(f"  Val   : {len(val_samples):>8,} → {val_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interim_dir", default="data/interim")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(Path(args.interim_dir), Path(args.out_dir), args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
