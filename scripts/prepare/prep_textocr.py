"""TextOCR — real natural images from Open Images with text annotations.

Dataset : ~28K images, 900K+ text instances
Source  : https://textvqa.org/textocr/dataset/
Output  : data/raw/textocr/<split>/<image_id>.jpg
          data/interim/textocr.jsonl
"""
import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm

ANNO_URLS = {
    "train": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_train.json",
    "val": "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json",
}


def _download_json(url: str, dest: Path) -> dict:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                bar.update(len(chunk))
    return json.loads(dest.read_text())


def _download_image(image_id: str, dest: Path) -> bool:
    if dest.exists():
        return True
    url = f"https://storage.googleapis.com/openimages/data/train/{image_id}.jpg"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            dest.write_bytes(r.content)
            return True
    except Exception:
        pass
    return False


def _annots_to_text(annot_ids: list[str], annots: dict) -> str:
    tokens = []
    entries = [(aid, annots[aid]) for aid in annot_ids if aid in annots]
    entries.sort(key=lambda x: (x[1]["bbox"][1], x[1]["bbox"][0]))
    for _, a in entries:
        txt = a.get("utf8_string", "")
        if txt and txt != ".":
            tokens.append(txt)
    return " ".join(tokens)


def run(raw_dir: Path, out_jsonl: Path) -> int:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for split, url in ANNO_URLS.items():
            cache = raw_dir / "textocr" / f"{split}.json"
            data = _download_json(url, cache)

            img_dir = raw_dir / "textocr" / split
            img_dir.mkdir(parents=True, exist_ok=True)

            annots = data.get("anns", {})
            imgs = data.get("imgs", {})

            for img_id, img_info in tqdm(imgs.items(), desc=f"TextOCR/{split}"):
                img_path = img_dir / f"{img_id}.jpg"
                if not _download_image(img_id, img_path):
                    continue

                annot_ids = data.get("imgToAnns", {}).get(img_id, [])
                text = _annots_to_text(annot_ids, annots)
                if not text.strip():
                    continue

                rel = str(img_path.relative_to(raw_dir))
                out_f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"TextOCR: {count} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/textocr.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
