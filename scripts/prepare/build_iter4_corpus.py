"""Assemble the iter-4 training corpus.

Mixes:
    - Filtered page pseudo-labels from label_pages_with_iter3.py
    - Word-level replay samples from prep_himalaya_indic.py output
      (prevents catastrophic forgetting on iter-3's word-level win)

Emits:
    - data/processed/train.jsonl
    - data/processed/val.jsonl        (5% pages held out; word replay all trains)

Ratio: default 5 pages : 1 word replay sample. Adjust via CLI.

The output schema matches what scripts/train.py expects (image, text,
prompt). Image paths are kept as-is — they're relative to their respective
raw_dir roots, which the trainer expects to see under a shared image_root.
For iter-4 that means both data/raw/indicdlp_pages/ and data/raw/himalaya_indic/
must be siblings under data/raw/ (they already are by default).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _filter_page_labels(records: list[dict]) -> list[dict]:
    """Keep only records that passed label_pages_with_iter3.py's quality gate."""
    return [r for r in records if r.get("filter") == "ok" and r.get("text")]


def _normalize_page(r: dict) -> dict:
    # Page images live under data/raw/indicdlp_pages/<lang>/...
    # Rewrite the image path to be relative to data/raw/ (the shared image_root)
    img = r["image"]
    if not img.startswith("indicdlp_pages/"):
        img = f"indicdlp_pages/{img}"
    return {
        "image":    img,
        "text":     r["text"],
        "prompt":   r.get("prompt",
                          "Extract this document page as clean markdown. "
                          "Preserve the original script exactly."),
        "source":   r.get("source", "indicdlp_iter3_bootstrap"),
    }


def _normalize_word(r: dict) -> dict:
    # Word images live under data/raw/himalaya_indic/... already
    return {
        "image":    r["image"],
        "text":     r["text"],
        "prompt":   r.get("prompt", "Transcribe the Devanagari text from this image:"),
        "source":   r.get("source", "himalaya_word_replay"),
    }


def run(
    labeled_pages_jsonl: Path,
    word_replay_jsonl: Path,
    train_out: Path,
    val_out: Path,
    max_pages: int,
    max_word_replay: int,
    val_ratio: float,
    seed: int,
) -> None:
    train_out.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Pages
    raw_pages = _load_jsonl(labeled_pages_jsonl)
    ok_pages = _filter_page_labels(raw_pages)
    rng.shuffle(ok_pages)
    if max_pages:
        ok_pages = ok_pages[:max_pages]
    print(f"Pages: {len(raw_pages):,} raw → {len(ok_pages):,} usable (filter=ok)")

    # Val split from pages only
    n_val = int(len(ok_pages) * val_ratio)
    val_pages   = ok_pages[:n_val]
    train_pages = ok_pages[n_val:]
    print(f"  → {len(train_pages):,} train / {len(val_pages):,} val (pages)")

    # Word replay — trains only, no val
    words = _load_jsonl(word_replay_jsonl)
    rng.shuffle(words)
    if max_word_replay:
        words = words[:max_word_replay]
    print(f"Word replay: {len(words):,} samples (all in train)")

    # Assemble
    train = [_normalize_page(r) for r in train_pages] + \
            [_normalize_word(r) for r in words]
    val   = [_normalize_page(r) for r in val_pages]
    rng.shuffle(train)

    with open(train_out, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_out, "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ratio_str = "n/a" if not words else f"{len(train_pages)/max(len(words),1):.1f}"
    print(f"\nWrote:")
    print(f"  train: {len(train):,} lines → {train_out}")
    print(f"    pages:{len(train_pages):,}  word_replay:{len(words):,}  "
          f"(ratio {ratio_str}:1)")
    print(f"  val:   {len(val):,} lines → {val_out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labeled_pages_jsonl", default="data/interim/indicdlp_labeled.jsonl")
    p.add_argument("--word_replay_jsonl",   default="data/interim/himalaya_indic.jsonl")
    p.add_argument("--train_out",           default="data/processed/train.jsonl")
    p.add_argument("--val_out",             default="data/processed/val.jsonl")
    p.add_argument("--max_pages",           type=int, default=2500)
    p.add_argument("--max_word_replay",     type=int, default=500)
    p.add_argument("--val_ratio",           type=float, default=0.05)
    p.add_argument("--seed",                type=int, default=42)
    args = p.parse_args()

    run(
        labeled_pages_jsonl=Path(args.labeled_pages_jsonl),
        word_replay_jsonl=Path(args.word_replay_jsonl),
        train_out=Path(args.train_out),
        val_out=Path(args.val_out),
        max_pages=args.max_pages,
        max_word_replay=args.max_word_replay,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
