"""FUNSD — real scanned English forms with text annotations.

Dataset : nielsr/funsd  (~200 samples)
Output  : data/raw/funsd/<split>/<idx>.png
          data/interim/funsd.jsonl
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _words_to_text(words: list[str], bboxes: list[list[int]]) -> str:
    if not words:
        return ""
    pairs = sorted(zip(words, bboxes), key=lambda p: (p[1][1], p[1][0]))
    return " ".join(w for w, _ in pairs if w.strip())


def run(raw_dir: Path, out_jsonl: Path) -> int:
    ds = load_dataset("nielsr/funsd")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for split in ["train", "test"]:
            img_dir = raw_dir / "funsd" / split
            img_dir.mkdir(parents=True, exist_ok=True)

            for idx, sample in enumerate(tqdm(ds[split], desc=f"FUNSD/{split}")):
                text = _words_to_text(sample["words"], sample["bboxes"])
                if not text.strip():
                    continue

                img_path = img_dir / f"{idx:05d}.png"
                if not img_path.exists():
                    sample["image"].save(img_path)

                rel = str(img_path.relative_to(raw_dir))
                f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"FUNSD: {count} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/funsd.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
