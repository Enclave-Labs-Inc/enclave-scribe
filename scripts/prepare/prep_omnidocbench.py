"""OmniDocBench — the primary benchmark for document OCR quality.

Dataset : opendatalab/OmniDocBench (HuggingFace)
Layout  : NOT a proper HF dataset — it's a raw file dump with a single
          `OmniDocBench.json` (all 1,651 annotations) and an `images/`
          directory of source pages. We download both and match locally.

Output  : data/raw/omnidocbench/images/<file>.png
          data/benchmark/omnidocbench_test.jsonl  (default — designed as held-out)

OmniDocBench is inherently a benchmark, not training data. By default
its entire 1,651-page set is routed to the held-out benchmark JSONL.

Categories (from page_attribute.data_source):
  book, academic_literature, colorful_textbook, note, PPT2PDF, ...
"""
import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm import tqdm


REPO_ID = "opendatalab/OmniDocBench"

# Layout detection categories we skip when building text:
#   abandon    — marked garbage
#   figure     — pure image, no OCR text
#   chart_mask — chart region, no text
SKIP_CATEGORIES = {"abandon", "figure", "chart_mask"}


def _layout_to_text(layout_dets: list[dict]) -> str:
    """Concatenate layout detections in reading order to reconstruct page text."""
    kept = []
    for det in layout_dets:
        if det.get("ignore", False):
            continue
        ct = det.get("category_type", "")
        if ct in SKIP_CATEGORIES:
            continue
        # Isolated equations use `latex`, everything else uses `text`
        content = det.get("latex", "") if ct == "equation_isolated" else det.get("text", "")
        content = (content or "").strip()
        if not content:
            continue
        order = det.get("order")
        # None-order items sink to the bottom (dataset uses None for extras)
        sort_key = order if isinstance(order, int) else 10**6
        kept.append((sort_key, content))
    kept.sort(key=lambda x: x[0])
    return "\n".join(text for _, text in kept)


def run(raw_dir: Path, benchmark_jsonl: Path) -> int:
    benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # 1. Download the annotations JSON
    print("Downloading OmniDocBench.json ...")
    ann_path = hf_hub_download(REPO_ID, "OmniDocBench.json", repo_type="dataset")

    # 2. Download all images (snapshot: batched, cached)
    print("Downloading OmniDocBench images (this can take a few minutes on first run) ...")
    img_dir = raw_dir / "omnidocbench"
    img_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir = snapshot_download(
        REPO_ID,
        repo_type="dataset",
        allow_patterns=["images/*"],
    )
    # Mirror images from HF cache into raw_dir/omnidocbench/images/
    src_img_dir = Path(snapshot_dir) / "images"
    dst_img_dir = img_dir / "images"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for src in src_img_dir.iterdir():
        dst = dst_img_dir / src.name
        if not dst.exists():
            # symlink is fine; falls back to copy if symlink fails on Windows/CI
            try:
                dst.symlink_to(src)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dst)

    # 3. Iterate annotations, build text, emit JSONL
    with open(ann_path) as f:
        data = json.load(f)

    count = 0
    missing = 0
    empty = 0
    category_counts: dict[str, int] = {}

    with open(benchmark_jsonl, "w", encoding="utf-8") as out_f:
        for item in tqdm(data, desc="OmniDocBench"):
            page = item.get("page_info", {})
            img_rel = page.get("image_path", "")
            if not img_rel:
                missing += 1
                continue

            # HF stores under images/<name>.png; strip any leading "images/"
            img_name = Path(img_rel).name
            img_file = dst_img_dir / img_name
            if not img_file.exists():
                missing += 1
                continue

            text = _layout_to_text(item.get("layout_dets", []))
            if not text.strip():
                empty += 1
                continue

            page_attr = page.get("page_attribute", {}) or {}
            data_source = page_attr.get("data_source", "unknown")
            category = f"omnidocbench_{data_source}"

            rel = str(img_file.relative_to(raw_dir))
            record = {
                "image": rel,
                "text": text,
                "category": category,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            category_counts[data_source] = category_counts.get(data_source, 0) + 1

    print(f"\nOmniDocBench: {count} samples (held out) → {benchmark_jsonl}")
    print(f"  Missing images: {missing}")
    print(f"  Empty text:     {empty}")
    print(f"  Per data_source:")
    for src, n in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"    {src:<25} {n:>5}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--benchmark_jsonl", default="data/benchmark/omnidocbench_test.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.benchmark_jsonl))


if __name__ == "__main__":
    main()
