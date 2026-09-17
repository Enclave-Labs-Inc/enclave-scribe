"""olmOCR-mix-1025 — full-page English OCR from AllenAI's own SFT corpus.

Dataset : allenai/olmOCR-mix-1025 (~270k pages, ODC-BY, GPT-4.1 transcriptions)
Output  : data/raw/olmocr_mix/<idx>.png
          data/interim/olmocr_mix.jsonl (training pool)

WHY THIS DATASET
    Iter-10 swaps base to Qwen3-VL-8B, giving up olmOCR-2's 270k-page English
    page-OCR pretraining. This dataset is *exactly* what olmOCR-2 was SFT'd on
    — borrowing their proven page-OCR corpus is the cleanest recovery.

    Composition (from HF dataset card):
      - 00_documents (232,790) — web-crawled PDFs, EN 94.46%
      - 01_books (17,474)      — Internet Archive, EN 91.28%
      - 02_loc_transcripts (9,989) — Library of Congress, EN 98.21%
      - 03_national_archives (9,997) — EN 99.82%

WHY THIS REWRITE (iter-12)
    Iter-11's version returned 0 samples silently: the real fields are
    `natural_text` / `pdf_relpath` / `page_number`, and images are NOT stored
    in the dataset — they must be rendered on demand from PDF tarballs under
    `pdf_tarballs/` in the HF repo. Rewrite loads the four subsets with
    weighted streaming, downloads the required tarballs, extracts the
    referenced PDFs, and renders the target pages with PyMuPDF.

USAGE
    python scripts/prepare/prep_olmocr_mix.py --limit 15000 --max_tarballs 20
"""
import argparse
import hashlib
import json
import random
import sys
import tarfile
import time
from pathlib import Path

from tqdm import tqdm


DATASET_ID = "allenai/olmOCR-mix-1025"

# Approximate subset sizes from the HF dataset card. Used only as sampling
# weights, so exact counts don't matter — just their ratios.
SUBSETS = {
    "00_documents":         232_790,
    "01_books":              17_474,
    "02_loc_transcripts":     9_989,
    "03_national_archives":   9_997,
}


def _tarball_for_relpath(pdf_relpath: str) -> str:
    """Return the tarball basename (with extension) that contains `pdf_relpath`.

    olmOCR-mix groups PDFs under `pdf_tarballs/` in the HF repo. The
    convention is that the first path component of `pdf_relpath` names the
    tarball (e.g. `pdf_relpath = "shard_00042/foo.pdf"` lives inside
    `pdf_tarballs/shard_00042.tar`). We treat the first component as the
    tarball basename and append `.tar`.
    """
    first = pdf_relpath.strip("/").split("/", 1)[0]
    if first.endswith(".tar") or first.endswith(".tar.gz"):
        return first
    return f"{first}.tar"


def _download_tarball(basename: str, cache_dir: Path, retries: int = 3) -> Path | None:
    """Download `pdf_tarballs/<basename>` from the HF repo with retries.

    The tarball lands at `cache_dir/pdf_tarballs/<basename>` (single on-disk
    location — hf_hub_download preserves the repo-relative path under
    `local_dir`). Multi-GB tarballs are not re-copied on rerun.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = cache_dir / "pdf_tarballs" / basename
    if local_path.exists():
        return local_path

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                repo_id=DATASET_ID,
                filename=f"pdf_tarballs/{basename}",
                repo_type="dataset",
                local_dir=str(cache_dir),
            )
            return Path(path)
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            last_err = e
            sleep = 2 ** attempt
            print(f"  tarball {basename} attempt {attempt + 1} failed: {e} (sleep {sleep}s)",
                  file=sys.stderr)
            time.sleep(sleep)
    print(f"  tarball {basename} FAILED after {retries} retries: {last_err}", file=sys.stderr)
    return None


def _extract_pdf(tarball_path: Path, pdf_relpath: str, pdf_cache_dir: Path) -> Path | None:
    """Extract the single PDF at `pdf_relpath` from `tarball_path`.

    Cached at `pdf_cache_dir/<sha1(pdf_relpath)>.pdf` so re-runs skip work.
    """
    pdf_cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(pdf_relpath.encode("utf-8")).hexdigest()[:16]
    out_path = pdf_cache_dir / f"{digest}.pdf"
    if out_path.exists():
        return out_path

    try:
        with tarfile.open(tarball_path, "r:*") as tar:
            # tar member names may be stored with or without the tarball-name
            # prefix; try both spellings.
            candidates = [pdf_relpath, pdf_relpath.split("/", 1)[-1]]
            member = None
            for name in candidates:
                try:
                    member = tar.getmember(name)
                    break
                except KeyError:
                    continue
            if member is None:
                return None
            fh = tar.extractfile(member)
            if fh is None:
                return None
            out_path.write_bytes(fh.read())
        return out_path
    except (tarfile.TarError, OSError) as e:
        print(f"  extract failed for {pdf_relpath}: {e}", file=sys.stderr)
        return None


def _render_page(pdf_path: Path, page_number: int, out_png: Path, dpi: int = 150) -> bool:
    """Render 1-indexed `page_number` of `pdf_path` to `out_png`."""
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  fitz.open failed for {pdf_path.name}: {e}", file=sys.stderr)
        return False
    try:
        page_idx = int(page_number) - 1
        if page_idx < 0 or page_idx >= len(doc):
            return False
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=dpi)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png))
        return True
    except Exception as e:
        print(f"  render failed for {pdf_path.name} p{page_number}: {e}", file=sys.stderr)
        return False
    finally:
        doc.close()


def _weighted_stream(seed: int):
    """Yield rows from the four subsets in weighted-random order."""
    from datasets import load_dataset

    iters = {}
    weights = []
    names = []
    for name, size in SUBSETS.items():
        ds = load_dataset(DATASET_ID, name=name, split="train", streaming=True)
        iters[name] = iter(ds)
        names.append(name)
        weights.append(size)

    rng = random.Random(seed)
    exhausted = set()
    while len(exhausted) < len(names):
        active = [(n, w) for n, w in zip(names, weights) if n not in exhausted]
        active_names, active_weights = zip(*active)
        pick = rng.choices(active_names, weights=active_weights, k=1)[0]
        try:
            row = next(iters[pick])
        except StopIteration:
            exhausted.add(pick)
            continue
        yield pick, row


def run(raw_dir: Path, out_jsonl: Path, limit: int, max_tarballs: int, seed: int) -> int:
    raw_dir = Path(raw_dir)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    base_dir = raw_dir / "olmocr_mix"
    tarball_dir = base_dir / "_tarballs"
    pdf_dir = base_dir / "_pdfs"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {DATASET_ID} (limit={limit:,}, max_tarballs={max_tarballs}, seed={seed}) ...",
          flush=True)

    downloaded_tarballs: dict[str, Path | None] = {}
    tarball_attempts = 0
    tarball_failures = 0

    kept = 0
    seen = 0
    pbar = tqdm(total=limit, desc="olmocr_mix")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for subset_name, sample in _weighted_stream(seed):
            if kept >= limit:
                break
            seen += 1

            text = sample.get("natural_text")
            text = str(text).strip() if text else ""
            if not text:
                continue

            pdf_relpath = sample.get("pdf_relpath")
            page_number = sample.get("page_number")
            if not pdf_relpath or page_number is None:
                continue

            tarball_name = _tarball_for_relpath(pdf_relpath)

            # Enforce the tarball-download cap before contacting HF.
            if tarball_name not in downloaded_tarballs:
                if len(downloaded_tarballs) >= max_tarballs:
                    # Skip rows that would require a new tarball beyond the cap.
                    continue
                tarball_attempts += 1
                path = _download_tarball(tarball_name, tarball_dir)
                downloaded_tarballs[tarball_name] = path
                if path is None:
                    tarball_failures += 1

            tarball_path = downloaded_tarballs[tarball_name]
            if tarball_path is None:
                continue

            pdf_path = _extract_pdf(tarball_path, pdf_relpath, pdf_dir)
            if pdf_path is None:
                continue

            img_path = base_dir / f"{kept:06d}.png"
            if not img_path.exists():
                if not _render_page(pdf_path, int(page_number), img_path):
                    continue

            rel = str(img_path.relative_to(raw_dir))
            f.write(json.dumps({
                "image":  rel,
                "text":   text,
                "source": "olmocr_mix_1025",
            }, ensure_ascii=False) + "\n")
            kept += 1
            pbar.update(1)

            if kept % 500 == 0:
                print(f"  kept={kept:,} seen={seen:,} subset_last={subset_name} "
                      f"tarballs={len(downloaded_tarballs)}", flush=True)

    pbar.close()

    if tarball_attempts > 0 and tarball_failures * 2 > tarball_attempts:
        raise RuntimeError(
            f"olmocr_mix tarball download failures {tarball_failures}/{tarball_attempts} "
            f"exceed 50% — aborting"
        )

    print(f"olmocr_mix: kept={kept:,} seen={seen:,} tarballs_used={len(downloaded_tarballs)} "
          f"→ {out_jsonl}")

    if kept == 0:
        raise RuntimeError("olmocr_mix produced 0 samples")

    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",       default="data/raw")
    parser.add_argument("--out_jsonl",     default="data/interim/olmocr_mix.jsonl")
    parser.add_argument("--limit",         type=int, default=15000)
    parser.add_argument("--max_tarballs",  type=int, default=20,
                        help="Hard cap on tarball downloads to bound HF egress cost")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.limit, args.max_tarballs, args.seed)


if __name__ == "__main__":
    main()
