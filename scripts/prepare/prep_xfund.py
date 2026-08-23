"""XFUND — real multilingual scanned forms (7 languages).

Languages : zh (Chinese), ja (Japanese), es (Spanish),
            fr (French),  it (Italian),  de (German), pt (Portuguese)
Source    : Direct download from Microsoft GitHub
Output    : data/raw/xfund/<lang>/<split>/<idx>.png
            data/interim/xfund.jsonl
"""
import argparse
import json
import os
import zipfile
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm

XFUND_BASE = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"
LANGUAGES = ["zh", "ja", "es", "fr", "it", "de", "pt"]


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def _words_to_text(document: list[dict]) -> str:
    tokens = []
    for item in document:
        for word in item.get("words", []):
            t = word.get("text", "").strip()
            if t:
                tokens.append(t)
    return " ".join(tokens)


def run(raw_dir: Path, out_jsonl: Path, benchmark_jsonl: Path | None = None) -> int:
    """XFUND ships each split as TWO files: a .zip of flat images and a
    separate .json of annotations. The old code only downloaded the .zip
    and looked for JSON inside it — that's why it produced 0 samples.

    val split is routed to benchmark_jsonl (held out) when provided.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    with open(out_jsonl, "w", encoding="utf-8") as train_f, \
         open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else _null_writer() as test_f:
        for lang in LANGUAGES:
            for split in ["train", "val"]:
                zip_name = f"{lang}.{split}.zip"
                json_name = f"{lang}.{split}.json"
                zip_path = raw_dir / "xfund" / zip_name
                json_path = raw_dir / "xfund" / json_name

                _download(f"{XFUND_BASE}/{zip_name}", zip_path)
                _download(f"{XFUND_BASE}/{json_name}", json_path)

                extract_dir = raw_dir / "xfund" / lang / split
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(extract_dir)

                data = json.loads(json_path.read_text(encoding="utf-8"))
                is_heldout = (split == "val" and benchmark_jsonl is not None)
                target = test_f if is_heldout else train_f

                for doc in data.get("documents", []):
                    img_file = extract_dir / doc["img"]["fname"]
                    if not img_file.exists():
                        continue

                    text = _words_to_text(doc.get("document", []))
                    if not text.strip():
                        continue

                    rel = str(img_file.relative_to(raw_dir))
                    record = {"image": rel, "text": text}
                    if is_heldout:
                        record["category"] = f"xfund_{lang}"
                    target.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if is_heldout:
                        test_count += 1
                    else:
                        train_count += 1

    print(f"XFUND: {train_count} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"XFUND: {test_count} val (held out) → {benchmark_jsonl}")
    return train_count + test_count


class _null_writer:
    """No-op file object stand-in for when benchmark_jsonl is not provided."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def write(self, _): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--out_jsonl",       default="data/interim/xfund.jsonl")
    parser.add_argument("--benchmark_jsonl", default="",
                        help="If set, val split (7 languages) is routed here as held-out test set")
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(Path(args.raw_dir), Path(args.out_jsonl), bench)


if __name__ == "__main__":
    main()
