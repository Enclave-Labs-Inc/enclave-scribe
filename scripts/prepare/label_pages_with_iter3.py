"""Bulk pseudo-label IndicDLP page images with iter-3 + agent.

Iter-4 hybrid bootstrap: IndicDLP gives us real Devanagari page images
but no OCR ground truth. This script generates the ground truth by
running iter-3 (loaded as a LoRA adapter over OLMoCR-2-7B) through the
agent's extract_page for each image. The bad_words_ids workaround in
scribe/agent/tools.py (PR #41) keeps output clean during labeling.

Pipeline:
    manifest of images  →  iter-3 extract_page  →  quality filter
                                                      ↓
                       data/interim/indicdlp_labeled.jsonl

Quality filters (drop the sample if ANY fires):
    1. len(text) < min_chars               → extraction failed
    2. ASCII ratio > max_ascii_ratio       → model hallucinated English
    3. any 20-char substring occurs 3+ x   → residual generation loop

Resume support: if the output JSONL exists, images already listed there
are skipped. Safe to interrupt and rerun.

Cost note: iter-3 inference is roughly 1-3 min per page on g5.xlarge.
For 3,000 images this is 50-150 GPU-hours (~$50-150). User approved
this upfront (see plan).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from scribe.agent import ModelRegistry
from scribe.agent import tools


# Default quality filter thresholds. Tuneable via CLI.
DEFAULT_MIN_CHARS       = 100
DEFAULT_MAX_ASCII_RATIO = 0.30
DEFAULT_LOOP_WINDOW     = 20    # substring length checked for repetition
DEFAULT_LOOP_MIN_HITS   = 3     # this many occurrences of the same 20-char slice = loop


def _load_done_stems(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["image"])
            except Exception:
                continue
    return done


def _ascii_ratio(s: str) -> float:
    if not s:
        return 1.0
    return sum(1 for c in s if ord(c) < 128) / len(s)


def _has_loop(s: str, window: int, min_hits: int) -> bool:
    """True if any `window`-char substring appears `min_hits`+ times."""
    if len(s) < window * min_hits:
        return False
    counts: dict[str, int] = {}
    for i in range(0, len(s) - window + 1):
        k = s[i:i+window]
        counts[k] = counts.get(k, 0) + 1
        if counts[k] >= min_hits:
            return True
    return False


def _quality_ok(text: str, min_chars: int, max_ascii: float,
                loop_window: int, loop_hits: int) -> tuple[bool, str]:
    if len(text) < min_chars:
        return False, "too_short"
    if _ascii_ratio(text) > max_ascii:
        return False, "too_ascii"
    if _has_loop(text, loop_window, loop_hits):
        return False, "repetition"
    return True, "ok"


def run(
    manifest_jsonl: Path,
    raw_dir: Path,
    out_jsonl: Path,
    base_model: str,
    adapter_dir: str,
    max_images: int,
    min_chars: int,
    max_ascii: float,
    loop_window: int,
    loop_hits: int,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest: list[dict] = []
    with open(manifest_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            manifest.append(json.loads(line))
    print(f"Manifest: {len(manifest):,} images from {manifest_jsonl}")

    # Resume
    done = _load_done_stems(out_jsonl)
    if done:
        print(f"Resume: {len(done):,} images already labeled, skipping those")
    todo = [m for m in manifest if m["image"] not in done]
    if max_images:
        todo = todo[:max_images]
    print(f"To label: {len(todo):,} images")

    if not todo:
        print("Nothing to do.")
        return

    # Load VLM once
    print(f"Loading VLM: {base_model} + adapter={adapter_dir or '<none>'} ...")
    registry = ModelRegistry(vlm_path=base_model, vlm_adapter_dir=adapter_dir)
    _ = registry.vlm()
    print(f"VLM ready on device={registry.vlm().device}")

    # Filter counters
    stats = {"ok": 0, "too_short": 0, "too_ascii": 0, "repetition": 0, "err": 0}

    t0 = time.time()
    with open(out_jsonl, "a", encoding="utf-8") as out_f:
        for m in tqdm(todo, desc="labeling"):
            rel = m["image"]
            img_path = raw_dir / rel
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                stats["err"] += 1
                out_f.write(json.dumps({
                    "image": rel, "error": f"image_open: {e}",
                }) + "\n")
                out_f.flush()
                continue

            try:
                text = tools.extract_page(img, registry)
            except Exception as e:
                stats["err"] += 1
                out_f.write(json.dumps({
                    "image": rel, "error": f"extract: {e}",
                }) + "\n")
                out_f.flush()
                continue

            ok, reason = _quality_ok(text, min_chars, max_ascii,
                                      loop_window, loop_hits)
            stats[reason] += 1
            record = {
                "image":    rel,
                "text":     text if ok else "",
                "language": m.get("language", ""),
                "source":   "indicdlp_iter3_bootstrap",
                "prompt":   "Extract this document page as clean markdown. "
                            "Preserve the original script exactly.",
                "filter":   reason,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    dt = time.time() - t0
    total = sum(stats.values())
    print(f"\nLabeled {total:,} images in {dt/60:.1f} min "
          f"({dt/max(total, 1):.1f}s/img)")
    print("Filter breakdown:")
    for k, v in stats.items():
        pct = 100 * v / max(total, 1)
        print(f"  {k:<12s}  {v:>6,d}  ({pct:.1f}%)")
    print(f"\nUsable (filter=ok) labels in {out_jsonl}: {stats['ok']:,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest_jsonl", default="data/interim/indicdlp_pages.manifest.jsonl")
    p.add_argument("--raw_dir",        default="data/raw/indicdlp_pages")
    p.add_argument("--out_jsonl",      default="data/interim/indicdlp_labeled.jsonl")
    p.add_argument("--base_model",     default="allenai/olmOCR-2-7B-1025")
    p.add_argument("--adapter_dir",    default="outputs/iter3")
    p.add_argument("--max_images",     type=int, default=0,
                   help="0 = all in manifest")
    p.add_argument("--min_chars",      type=int, default=DEFAULT_MIN_CHARS)
    p.add_argument("--max_ascii",      type=float, default=DEFAULT_MAX_ASCII_RATIO)
    p.add_argument("--loop_window",    type=int, default=DEFAULT_LOOP_WINDOW)
    p.add_argument("--loop_hits",      type=int, default=DEFAULT_LOOP_MIN_HITS)
    args = p.parse_args()

    run(
        manifest_jsonl=Path(args.manifest_jsonl),
        raw_dir=Path(args.raw_dir),
        out_jsonl=Path(args.out_jsonl),
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        max_images=args.max_images,
        min_chars=args.min_chars,
        max_ascii=args.max_ascii,
        loop_window=args.loop_window,
        loop_hits=args.loop_hits,
    )


if __name__ == "__main__":
    main()
