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
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from scribe.eval.metrics import compute_all

_first_exception_logged = False

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


def _run_sample(
    model,
    image_root: Path,
    sample: dict,
    default_prompt: str,
    max_new_tokens: int,
    per_category_max_new_tokens: dict[str, int] | None = None,
) -> dict:
    """Run inference on one sample. Honors per-sample `prompt` field and
    per-category `max_new_tokens` overrides.

    Per-sample prompt routing: OCRBench V2 (and DocVQA, and any prompt-per-
    sample benchmark) carries the actual task text in `sample["prompt"]`;
    for those benchmarks a fixed prompt like "document parsing." is wrong
    (iter-6 flagged this; iter-7a inherited the bug). We now pass
    `sample.get("prompt", default_prompt)` through to `infer_image`, so
    each benchmark gets the prompt its authors designed.

    Per-category token cap: OmniDocBench's `research_report` samples are
    long dense pages that overflow the default 768-token cap and produce
    truncated F1=0.07 numbers. Passing `{"omnidocbench_research_report":
    4096}` (or similar) via `--per_category_max_new_tokens` uncaps just
    those categories without inflating cost across the whole run.
    """
    global _first_exception_logged
    from scribe.infer.local import infer_image
    image_path = str(image_root / sample["image"]) if image_root else sample["image"]
    prompt_used = sample.get("prompt", default_prompt)
    category = sample.get("category", "all")
    tokens_used = (per_category_max_new_tokens or {}).get(category, max_new_tokens)
    try:
        pred = infer_image(model, image_path, prompt=prompt_used, max_new_tokens=tokens_used)
    except Exception as e:
        # Surface the first exception per run to stderr so silent 100%-CER
        # runs are easy to diagnose. Subsequent exceptions are still swallowed
        # so a single bad sample doesn't spam the log across 500 samples.
        pred = ""
        if not _first_exception_logged:
            _first_exception_logged = True
            print(f"\n[eval.py] first per-sample exception on {sample.get('image')!r}: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("[eval.py] further per-sample exceptions are swallowed silently.", file=sys.stderr)
    metrics = compute_all(pred, sample["text"])
    return {
        "image":       sample["image"],
        "category":    category,
        "prompt_used": prompt_used,
        "tokens_cap":  tokens_used,
        "metrics":     metrics,
        "pred":        pred,
        "gt":          sample["text"],
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

    # Parse per-category max_new_tokens override (JSON: {"cat": int, ...}).
    # Silently accepts {} / None; malformed input errors up front rather than mid-run.
    per_cat_caps: dict[str, int] | None = None
    if args.per_category_max_new_tokens:
        per_cat_caps = json.loads(args.per_category_max_new_tokens)
        if not isinstance(per_cat_caps, dict) or any(not isinstance(v, int) for v in per_cat_caps.values()):
            raise ValueError("--per_category_max_new_tokens must be JSON dict {str: int}")
        if per_cat_caps:
            print(f"Per-category max_new_tokens overrides: {per_cat_caps}")

    results = []
    t0 = time.time()
    for sample in tqdm(samples, desc="Evaluating"):
        results.append(_run_sample(
            model, image_root, sample,
            default_prompt=args.default_prompt,
            max_new_tokens=args.max_new_tokens,
            per_category_max_new_tokens=per_cat_caps,
        ))

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
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Per-sample generation budget (default 4096). "
                             "Lower for eval on page-level inputs where full-markdown "
                             "generation is too slow (e.g. 512).")
    parser.add_argument("--default_prompt", default="document parsing.",
                        help="Fallback prompt when a sample has no 'prompt' field. "
                             "Per-sample prompts (from the JSONL) always take precedence — "
                             "e.g. OCRBench V2's per-task questions are used verbatim.")
    parser.add_argument("--per_category_max_new_tokens", default="",
                        help="JSON dict of category -> int, overrides --max_new_tokens for "
                             "specific categories. Example: "
                             "'{\"omnidocbench_research_report\": 4096}' uncaps long research "
                             "pages without inflating cost across the run.")
    main_args = parser.parse_args()
    run(main_args)


if __name__ == "__main__":
    main()
