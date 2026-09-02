"""ai4bharat/indicdlp — page images for iter-4 hybrid bootstrap.

Iter-4 uses this dataset as the SOURCE of real page images. It ships with
119k human-annotated Indic document pages across 12 languages (163 train
shards). We only care about the Devanagari-script subset (Hindi + Marathi).

The dataset was built for LAYOUT parsing (COCO-style bboxes over regions),
not OCR. Text ground truth comes from iter-3 in the next step
(label_pages_with_iter3.py). This script just extracts images.

Access:
- MIT license but GATED — accept terms at:
    https://huggingface.co/datasets/ai4bharat/indicdlp
- Then our HF token can download.

Discovered dataset structure (2026-09-02, direct probe):
- 163 train parquet shards (~500 MB each), each shard is single-language
- Language is encoded in image.path as: <doctype>_<lang>_<id>_<sub>.png
    (e.g. "qp_hi_000133_0.png" = Question Paper, Hindi, sample 133, page 0)
- Shards are alphabetically ordered by language. Confirmed:
    shard  0 = as (Assamese)
    shard 52 = hi (Hindi)
    shard 91 = mr (Marathi)
- datasets.load_dataset(streaming=True) HANGS on this gated dataset
  during shard enumeration. This script uses direct hf_hub_download of
  individual shards instead.

Strategy: exploit the alphabetical shard order. Start at a known-good
first shard per target language (from LANG_START_SHARD), then walk
forward one shard at a time, verifying first-row language. Stop when:
    (a) shard language no longer matches (block end), OR
    (b) per-language sample cap hit.

This avoids downloading unrelated shards. For Hindi + Marathi at ~13
shards each, expect ~26 downloads (~13 GB), not 163 (~82 GB).

Cache management: downloaded shards are large. We DELETE the cached
parquet after extraction (unless --keep_cache) to avoid exhausting disk
during Phase 2's labeling step which shares the same volume.

Output:
- data/raw/indicdlp_pages/<lang>/<shard_idx>/<stem>.png
- data/interim/indicdlp_pages.manifest.jsonl — one line per image:
    {"image": "<lang>/<shard_idx>/<stem>.png", "language": "hi",
     "source": "indicdlp", "shard": 52, "orig": "qp_hi_000133_0.png"}
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm


DATASET_ID = "ai4bharat/indicdlp"
DEFAULT_LANGS = ("hi", "mr")

# Known first-shard index per language (verified 2026-09-02).
# ai4bharat/indicdlp bundles shards alphabetically by ISO language code.
# If a language isn't in this map, prep script errors — add it here.
LANG_START_SHARD = {
    "as": 0,     # Assamese  (verified)
    "hi": 52,    # Hindi     (verified)
    "mr": 91,    # Marathi   (verified)
    # Others deducible from alphabetical order with ~13 shards each:
    # bn ≈ 13, en ≈ 26, gu ≈ 39, kn ≈ 65, ml ≈ 78, or ≈ 104,
    # pa ≈ 117, sa ≈ 130 (if present), ta ≈ 143, te ≈ 156.
    # Adding to LANG_START_SHARD isn't strictly required — the script
    # verifies via first-row detection and errors clearly if wrong.
}

# doctype_lang_id_sub.png (also handles doctype_lang_id.png without _sub)
_PATH_LANG_RE = re.compile(r"^[a-z]{2}_([a-z]{2})_\d+(?:_\d+)?\.png$")


def _list_train_shards(token: str) -> list[str]:
    api = HfApi(token=token)
    files = api.list_repo_files(DATASET_ID, repo_type="dataset")
    return sorted(
        f for f in files
        if f.startswith("data/train-") and f.endswith(".parquet")
    )


def _detect_shard_lang(parquet_path: Path) -> str | None:
    pf = pq.ParquetFile(str(parquet_path))
    try:
        batch = next(pf.iter_batches(batch_size=1, columns=["image"]))
    except StopIteration:
        return None
    rows = batch.to_pylist()
    if not rows:
        return None
    p = rows[0]["image"].get("path", "")
    m = _PATH_LANG_RE.match(str(p))
    return m.group(1) if m else None


def _extract_shard_images(
    parquet_path: Path, lang: str, shard_idx: int, raw_dir: Path,
    manifest_f, max_remaining: int,
) -> int:
    out_dir = raw_dir / lang / f"{shard_idx:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(str(parquet_path))
    saved = 0
    for batch in pf.iter_batches(batch_size=32, columns=["image"]):
        for r in batch.to_pylist():
            if saved >= max_remaining:
                return saved
            img_bytes = r["image"].get("bytes")
            img_path  = r["image"].get("path", "")
            if not img_bytes:
                continue
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception:
                continue
            stem = Path(str(img_path)).stem or f"{saved:05d}"
            out = out_dir / f"{stem}.png"
            img.save(out, "PNG")
            manifest_f.write(json.dumps({
                "image":    str(out.relative_to(raw_dir)),
                "language": lang,
                "source":   "indicdlp",
                "shard":    shard_idx,
                "orig":     img_path,
            }) + "\n")
            manifest_f.flush()
            saved += 1
    return saved


def _drop_cached_parquet(parquet_path: Path) -> None:
    """Delete downloaded parquet + its HF blob (they're symlinked)."""
    try:
        if parquet_path.is_symlink():
            blob = parquet_path.resolve()
            parquet_path.unlink(missing_ok=True)
            blob.unlink(missing_ok=True)
        else:
            parquet_path.unlink(missing_ok=True)
    except Exception:
        pass


def _pull_shard(shards: list[str], idx: int, token: str) -> Path:
    return Path(hf_hub_download(
        DATASET_ID, shards[idx], repo_type="dataset", token=token,
    ))


def run_one_lang(
    shards: list[str], lang: str, start_idx: int,
    raw_dir: Path, manifest_f, max_samples: int,
    keep_cache: bool, token: str,
) -> int:
    """Walk forward from start_idx until the language changes or cap hit."""
    if start_idx >= len(shards):
        print(f"[{lang}] start index {start_idx} beyond {len(shards)} shards")
        return 0

    saved = 0
    idx = start_idx
    while saved < max_samples and idx < len(shards):
        try:
            pq_path = _pull_shard(shards, idx, token)
        except Exception as e:
            print(f"[{lang} shard {idx}] download failed: {e}")
            idx += 1
            continue

        detected = _detect_shard_lang(pq_path)
        if detected != lang:
            print(f"[{lang} shard {idx}] boundary reached — detected '{detected}' "
                  f"(not '{lang}'). Stopping.")
            if not keep_cache:
                _drop_cached_parquet(pq_path)
            break

        cap = max_samples - saved
        n = _extract_shard_images(pq_path, lang, idx, raw_dir, manifest_f, cap)
        saved += n
        print(f"[{lang} shard {idx}] +{n} images  (lang total {saved}/{max_samples})")

        if not keep_cache:
            _drop_cached_parquet(pq_path)
        idx += 1

    return saved


def run(
    raw_dir: Path, manifest_jsonl: Path, langs: tuple[str, ...],
    max_per_lang: int, keep_cache: bool, token: str,
) -> int:
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    shards = _list_train_shards(token)
    print(f"IndicDLP: {len(shards)} train shards. Target langs: {langs}")

    missing = [l for l in langs if l not in LANG_START_SHARD]
    if missing:
        raise SystemExit(
            f"Unknown start shard for language(s) {missing}. "
            f"Verify the shard index by downloading and checking, then add to "
            f"LANG_START_SHARD in this file. Known: {sorted(LANG_START_SHARD)}"
        )

    total = 0
    per_lang = {}
    with open(manifest_jsonl, "a", encoding="utf-8") as mf:
        for lang in langs:
            n = run_one_lang(
                shards, lang, LANG_START_SHARD[lang],
                raw_dir, mf, max_per_lang, keep_cache, token,
            )
            per_lang[lang] = n
            total += n

    print()
    print(f"Done. Wrote {total:,} images across languages:")
    for l, n in per_lang.items():
        print(f"  {l}: {n:,}")
    print(f"Manifest → {manifest_jsonl}")
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",         default="data/raw/indicdlp_pages")
    p.add_argument("--manifest_jsonl",  default="data/interim/indicdlp_pages.manifest.jsonl")
    p.add_argument("--langs",           nargs="+", default=list(DEFAULT_LANGS))
    p.add_argument("--max_per_lang",    type=int, default=1_800,
                   help="Per-language sample cap. Default 1800 → 3600 total for hi+mr")
    p.add_argument("--keep_cache",      action="store_true",
                   help="Keep downloaded parquets (~500 MB each). Default: delete after use.")
    p.add_argument("--hf_token_env",    default="HF_TOKEN")
    args = p.parse_args()

    token = os.environ.get(args.hf_token_env)
    if not token:
        raise SystemExit(f"Set {args.hf_token_env}=<hf_...> "
                         "and accept ai4bharat/indicdlp terms on HF first")

    existing = 0
    mf_path = Path(args.manifest_jsonl)
    if mf_path.exists():
        existing = sum(1 for _ in open(mf_path))
        print(f"NOTE: manifest already has {existing:,} entries — this run APPENDS.")

    run(
        raw_dir=Path(args.raw_dir),
        manifest_jsonl=mf_path,
        langs=tuple(l.lower() for l in args.langs),
        max_per_lang=args.max_per_lang,
        keep_cache=args.keep_cache,
        token=token,
    )


if __name__ == "__main__":
    main()
