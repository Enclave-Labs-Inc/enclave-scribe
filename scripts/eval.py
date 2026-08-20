"""EnclaveScribe — comprehensive evaluation against any ground-truth JSONL.

Computes per-sample and per-category: NED, CER, WER, BLEU-4, Token F1.
Outputs a results JSON + a printed comparison table.

Usage:
    # Evaluate on OmniDocBench (primary benchmark)
    python scripts/eval.py \
        --gt_jsonl    data/benchmark/omnidocbench.jsonl \
        --image_root  data/raw \
        --model_dir   outputs/lora_lambda_2xa100 \
        --out_json    results/omnidocbench.json

    # Evaluate on custom JSONL
    python scripts/eval.py \
        --gt_jsonl   data/processed/val.jsonl \
        --image_root data/raw \
        --model_dir  outputs/lora_lambda_2xa100
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scribe.eval.metrics import compute_all

# Published scores for comparison (OmniDocBench, lower NED = better)
SOTA_COMPARISON = {
    "Unlimited-OCR":   {"ned": 0.082, "f1": 0.921},
    "DocOwl-1.5":      {"ned": 0.198, "f1": 0.810},
    "TextMonkey":      {"ned": 0.215, "f1": 0.793},
    "GOT-OCR2.0":      {"ned": 0.143, "f1": 0.867},
    "InternVL2-8B":    {"ned": 0.175, "f1": 0.835},
    "Qwen2.5-VL-7B":   {"ned": 0.131, "f1": 0.878},
}


def _load_model(base_model: str, adapter_dir: str = ""):
    from scribe.model.vlm import Qwen2VLModel
    model = Qwen2VLModel()
    model.load(base_model, adapter_dir=adapter_dir)
    return model


def _run_sample(model, image_root: Path, sample: dict) -> dict:
    from scribe.infer.local import infer_image
    image_path = str(image_root / sample["image"]) if image_root else sample["image"]
    try:
        pred = infer_image(model, image_path)
    except Exception as e:
        pred = ""
    metrics = compute_all(pred, sample["text"])
    return {
        "image":    sample["image"],
        "category": sample.get("category", "all"),
        "metrics":  metrics,
        "pred":     pred,
        "gt":       sample["text"],
    }


def _print_table(rows: list[tuple], title: str, headers: list[str]):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in col_w)
    header = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print(f"\n{'─' * len(header)}")
    print(title)
    print(f"{'─' * len(header)}")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(str(v).ljust(col_w[i]) for i, v in enumerate(row)))


def run(args):
    samples = [json.loads(l) for l in open(args.gt_jsonl, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[:args.limit]

    image_root = Path(args.image_root) if args.image_root else None
    label = f"{args.base_model} + {args.adapter_dir}" if args.adapter_dir else args.base_model
    print(f"Evaluating {len(samples)} samples from {args.gt_jsonl}")
    print(f"Model: {label}\n")

    model = _load_model(args.base_model, adapter_dir=args.adapter_dir)

    results = []
    t0 = time.time()
    for sample in tqdm(samples, desc="Evaluating"):
        results.append(_run_sample(model, image_root, sample))

    elapsed = time.time() - t0
    print(f"\nInference: {elapsed:.1f}s for {len(results)} samples ({elapsed/len(results):.2f}s/sample)")

    # Aggregate by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["metrics"])
        by_cat["__all__"].append(r["metrics"])

    def avg(metric: str, items: list[dict]) -> float:
        return round(sum(x[metric] for x in items) / max(len(items), 1), 4)

    # Per-category table
    cat_rows = []
    for cat, items in sorted(by_cat.items()):
        label = "OVERALL" if cat == "__all__" else cat
        cat_rows.append((
            label,
            len(items),
            avg("ned", items),
            avg("cer", items),
            avg("wer", items),
            avg("bleu", items),
            avg("f1", items),
        ))
    cat_rows.sort(key=lambda r: (r[0] == "OVERALL", r[0]))

    _print_table(
        cat_rows,
        "EnclaveScribe — Results by Category",
        ["Category", "N", "NED↓", "CER↓", "WER↓", "BLEU↑", "F1↑"],
    )

    # Comparison vs SOTA — only meaningful on OmniDocBench
    overall = by_cat["__all__"]
    our_ned = avg("ned", overall)
    our_f1 = avg("f1", overall)

    if "omnidocbench" in args.gt_jsonl.lower():
        comp_rows = [("EnclaveScribe", f"{our_ned:.3f}", f"{our_f1:.3f}", "← ours")]
        for model_name, scores in sorted(SOTA_COMPARISON.items(), key=lambda x: x[1]["ned"]):
            delta_ned = our_ned - scores["ned"]
            flag = "✓ better" if delta_ned < 0 else "✗ behind"
            comp_rows.append((model_name, f"{scores['ned']:.3f}", f"{scores['f1']:.3f}", flag))
        _print_table(
            comp_rows,
            "Comparison vs Published Models (OmniDocBench, NED lower is better)",
            ["Model", "NED↓", "F1↑", "Status"],
        )

    # Save full results
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": label,
            "benchmark": args.gt_jsonl,
            "n_samples": len(results),
            "elapsed_s": round(elapsed, 2),
            "overall": {m: avg(m, overall) for m in ("ned", "cer", "wer", "bleu", "f1")},
            "by_category": {
                cat: {m: avg(m, items) for m in ("ned", "cer", "wer", "bleu", "f1")}
                for cat, items in by_cat.items()
            },
            "samples": results,
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nFull results saved → {args.out_json}")


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe evaluation")
    parser.add_argument("--gt_jsonl",    required=True, help="Ground truth JSONL")
    parser.add_argument("--image_root",  default="data/raw")
    parser.add_argument("--base_model",  default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="", help="Path to LoRA adapter (empty = base model only)")
    parser.add_argument("--out_json",    default="results/eval.json")
    parser.add_argument("--limit",       type=int, default=0, help="Limit samples (0 = all)")
    main_args = parser.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
