"""CORD v2 — real receipt images with structured text.

Dataset : naver-clova-ix/cord-v2  (~11K samples, Korean + English)
Output  : data/raw/cord/<split>/<idx>.png
          data/interim/cord.jsonl
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _parse_ground_truth(gt_str: str) -> str:
    try:
        gt = json.loads(gt_str)
    except Exception:
        return ""

    lines = []
    parse = gt.get("gt_parse", {})

    for item in parse.get("menu", []):
        if not isinstance(item, dict):
            continue
        parts = [item.get("nm", ""), item.get("cnt", ""), item.get("price", "")]
        row = "  ".join(p for p in parts if p)
        if row:
            lines.append(row)

    sub = parse.get("sub_total", {})
    if isinstance(sub, dict):
        if sub.get("subtotal_price"):
            lines.append(f"Subtotal: {sub['subtotal_price']}")
        if sub.get("tax_price"):
            lines.append(f"Tax: {sub['tax_price']}")
        if sub.get("discount_price"):
            lines.append(f"Discount: {sub['discount_price']}")

    total = parse.get("total", {})
    if isinstance(total, dict):
        if total.get("total_price"):
            lines.append(f"Total: {total['total_price']}")
        if total.get("cashprice"):
            lines.append(f"Cash: {total['cashprice']}")
        if total.get("changeprice"):
            lines.append(f"Change: {total['changeprice']}")

    return "\n".join(lines)


def run(raw_dir: Path, out_jsonl: Path) -> int:
    ds = load_dataset("naver-clova-ix/cord-v2")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for split in ["train", "validation", "test"]:
            img_dir = raw_dir / "cord" / split
            img_dir.mkdir(parents=True, exist_ok=True)

            for idx, sample in enumerate(tqdm(ds[split], desc=f"CORD/{split}")):
                text = _parse_ground_truth(sample["ground_truth"])
                if not text.strip():
                    continue

                img_path = img_dir / f"{idx:05d}.png"
                if not img_path.exists():
                    sample["image"].save(img_path)

                rel = str(img_path.relative_to(raw_dir))
                f.write(json.dumps({"image": rel, "text": text}, ensure_ascii=False) + "\n")
                count += 1

    print(f"CORD: {count} samples → {out_jsonl}")
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_jsonl", default="data/interim/cord.jsonl")
    args = parser.parse_args()
    run(Path(args.raw_dir), Path(args.out_jsonl))


if __name__ == "__main__":
    main()
