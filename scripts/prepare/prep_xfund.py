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


def run(raw_dir: Path, out_jsonl: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for lang in LANGUAGES:
            for split in ["train", "val"]:
                zip_name = f"{lang}.{split}.zip"
                zip_url = f"{XFUND_BASE}/{zip_name}"
                zip_path = raw_dir / "xfund" / zip_name

                _download(zip_url, zip_path)

                extract_dir = raw_dir / "xfund" / lang / split
                extract_dir.mkdir(parents=True, exist_ok=True)

                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(extract_dir)

                json_files = list(extract_dir.rglob("*.json"))
                if not json_files:
                    print(f"WARNING: no JSON found for {lang}/{split}")
                    continue

                for json_file in json_files:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    for doc in data.get("documents", []):
                        img_file = json_file.parent / doc["img"]["fname"]
                        if not img_file.exists():
                            continue

                        text = _words_to_text(doc.get("document", []))
                        if not text.strip():
                            continue

                        rel = str(img_file.relative_to(raw_dir))
                        f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                        count += 1

    print(f"XFUND: {count} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/xfund.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
