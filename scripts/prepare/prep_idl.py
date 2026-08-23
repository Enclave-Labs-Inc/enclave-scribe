"""IDL-WDS — real industry document pages (tobacco, pharma, legal).

Dataset : pixparse/idl-wds on HuggingFace (WebDataset format, ~1-10M pages)
          We sample a configurable subset for training.
Output  : data/raw/idl/<shard_idx>/<idx>.png
          data/interim/idl.jsonl

Sample schema (per pixparse/idl-wds card):
  __key__, __url__  : string identifiers
  pdf               : bytes — original PDF
  tif               : bytes — TIFF rendering (possibly multi-page)
  json              : dict  — Textract OCR with pages: [{text: [...], bbox, poly, score}]
  ocr               : bytes — legacy OCR (ignored, json is newer)

We render the TIFF's pages one-at-a-time and pair each page with the
corresponding json['pages'][i]['text'] list. Each PAGE becomes one
training sample (not each document), giving finer-grained supervision.
"""
import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageSequence
from tqdm import tqdm


DATASET_ID = "pixparse/idl-wds"
SHARD_SIZE = 5000


def _tiff_frames(tif_bytes: bytes):
    """Yield each frame in a multi-page TIFF as an RGB PIL image."""
    with Image.open(io.BytesIO(tif_bytes)) as img:
        for frame in ImageSequence.Iterator(img):
            yield frame.convert("RGB")


def _page_text(page: dict) -> str:
    """Concatenate a page's text lines in reading order."""
    lines = page.get("text", []) or []
    return "\n".join(line for line in lines if line and line.strip())


def _parse_json_field(raw) -> dict | None:
    """The json field may arrive as dict, bytes, or str depending on codec."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return None
    return None


def run(raw_dir: Path, out_jsonl: Path, max_samples: int = 100_000) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_ID} (target {max_samples:,} PAGES)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    img_base = raw_dir / "idl"
    img_base.mkdir(parents=True, exist_ok=True)

    count = 0
    docs_seen = 0
    docs_skipped = 0
    shard = 0
    shard_dir = img_base / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        pbar = tqdm(total=max_samples, desc="IDL pages")
        for sample in ds:
            if count >= max_samples:
                break
            docs_seen += 1

            tif_bytes = sample.get("tif")
            if not tif_bytes or not isinstance(tif_bytes, (bytes, bytearray)):
                docs_skipped += 1
                continue

            ann = _parse_json_field(sample.get("json"))
            if not ann:
                docs_skipped += 1
                continue

            pages_ann = ann.get("pages", []) or []
            if not pages_ann:
                docs_skipped += 1
                continue

            try:
                frames = list(_tiff_frames(tif_bytes))
            except Exception:
                docs_skipped += 1
                continue

            # Pair frames with page annotations by index; skip mismatched extras
            for i, frame in enumerate(frames):
                if count >= max_samples:
                    break
                if i >= len(pages_ann):
                    break

                text = _page_text(pages_ann[i])
                if not text.strip() or len(text) < 20:
                    continue

                # Rotate to new shard every SHARD_SIZE samples
                if count > 0 and count % SHARD_SIZE == 0:
                    shard += 1
                    shard_dir = img_base / f"{shard:04d}"
                    shard_dir.mkdir(exist_ok=True)

                img_path = shard_dir / f"{count % SHARD_SIZE:05d}.png"
                frame.save(img_path, "PNG", optimize=True)

                rel = str(img_path.relative_to(raw_dir))
                out_f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                count += 1
                pbar.update(1)

        pbar.close()

    print(f"IDL-WDS: {count:,} page samples from {docs_seen:,} documents → {out_jsonl}")
    print(f"  Skipped (missing tif/json/empty pages): {docs_skipped:,} documents")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",     default="data/raw")
    parser.add_argument("--out_jsonl",   default="data/interim/idl.jsonl")
    parser.add_argument("--max_samples", type=int, default=100_000,
                        help="Cap on total PAGES emitted (not documents)")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl), args.max_samples)


if __name__ == "__main__":
    main()
