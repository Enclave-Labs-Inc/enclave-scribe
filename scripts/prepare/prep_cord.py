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


def _flatten(v) -> str:
    """CORD fields can be str, list of str, or nested — normalise to a single string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return " ".join(_flatten(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_flatten(x) for x in v.values())
    return str(v)


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
        parts = [_flatten(item.get("nm")), _flatten(item.get("cnt")), _flatten(item.get("price"))]
        row = "  ".join(p for p in parts if p)
        if row:
            lines.append(row)

    sub = parse.get("sub_total", {})
    if isinstance(sub, dict):
        if sub.get("subtotal_price"):
            lines.append(f"Subtotal: {_flatten(sub['subtotal_price'])}")
        if sub.get("tax_price"):
            lines.append(f"Tax: {_flatten(sub['tax_price'])}")
        if sub.get("discount_price"):
            lines.append(f"Discount: {_flatten(sub['discount_price'])}")

    total = parse.get("total", {})
    if isinstance(total, dict):
        if total.get("total_price"):
            lines.append(f"Total: {_flatten(total['total_price'])}")
        if total.get("cashprice"):
            lines.append(f"Cash: {_flatten(total['cashprice'])}")
        if total.get("changeprice"):
            lines.append(f"Change: {_flatten(total['changeprice'])}")

    return "\n".join(lines)


def run(raw_dir: Path, out_jsonl: Path, benchmark_jsonl: Path | None = None) -> int:
    """Route CORD splits: train+validation → out_jsonl (training pool);
    test → benchmark_jsonl (held-out test set), if provided.
    Falls back to old behavior (all splits → out_jsonl) when benchmark_jsonl is None.
    """
    ds = load_dataset("naver-clova-ix/cord-v2")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count = 0
    with open(out_jsonl, "w", encoding="utf-8") as train_f, \
         open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else _null_writer() as test_f:
        for split in ["train", "validation", "test"]:
            img_dir = raw_dir / "cord" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            is_heldout = (split == "test" and benchmark_jsonl is not None)
            target = test_f if is_heldout else train_f

            for idx, sample in enumerate(tqdm(ds[split], desc=f"CORD/{split}")):
                text = _parse_ground_truth(sample["ground_truth"])
                if not text.strip():
                    continue

                img_path = img_dir / f"{idx:05d}.png"
                if not img_path.exists():
                    sample["image"].save(img_path)

                rel = str(img_path.relative_to(raw_dir))
                record = {"image": rel, "text": text}
                if is_heldout:
                    record["category"] = "cord"
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                if is_heldout:
                    test_count += 1
                else:
                    train_count += 1

    print(f"CORD: {train_count} train/val → {out_jsonl}")
    if benchmark_jsonl:
        print(f"CORD: {test_count} test → {benchmark_jsonl} (held out)")
    return train_count + test_count


class _null_writer:
    """No-op file object stand-in for when benchmark_jsonl is not provided."""
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def write(self, _): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",          default="data/raw")
    parser.add_argument("--out_jsonl",        default="data/interim/cord.jsonl")
    parser.add_argument("--benchmark_jsonl",  default="",
                        help="If set, official test split is routed here instead of --out_jsonl")
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(Path(args.raw_dir), Path(args.out_jsonl), bench)


if __name__ == "__main__":
    main()
