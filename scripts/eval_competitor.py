"""Run any OpenAI-compatible model on an OCR benchmark and compute scores.

Used to benchmark EnclaveScribe against:
  - Interfaze.ai  (OCRBench V2: 70.7%)
  - Gemini (via OpenAI-compat endpoint)
  - Claude  (via Anthropic SDK mapped to OpenAI compat)
  - GPT-4o / GPT-5
  - Grok

Usage:
    # Interfaze.ai
    python scripts/eval_competitor.py \\
        --gt_jsonl  data/benchmark/ocrbench_v2.jsonl \\
        --image_root data/raw \\
        --api_base  https://api.interfaze.ai/v1 \\
        --api_key   $INTERFAZE_API_KEY \\
        --model     interfaze-ocr \\
        --out_json  results/competitor_interfaze.json

    # GPT-4o
    python scripts/eval_competitor.py \\
        --gt_jsonl  data/benchmark/ocrbench_v2.jsonl \\
        --api_base  https://api.openai.com/v1 \\
        --api_key   $OPENAI_API_KEY \\
        --model     gpt-4o \\
        --out_json  results/competitor_gpt4o.json

    # Claude (via Anthropic OpenAI-compat layer)
    python scripts/eval_competitor.py \\
        --gt_jsonl  data/benchmark/ocrbench_v2.jsonl \\
        --api_base  https://api.anthropic.com/v1 \\
        --api_key   $ANTHROPIC_API_KEY \\
        --model     claude-sonnet-4-5 \\
        --out_json  results/competitor_claude.json

    # Gemini (via Google OpenAI-compat endpoint)
    python scripts/eval_competitor.py \\
        --gt_jsonl  data/benchmark/ocrbench_v2.jsonl \\
        --api_base  https://generativelanguage.googleapis.com/v1beta/openai \\
        --api_key   $GOOGLE_API_KEY \\
        --model     gemini-2.5-flash \\
        --out_json  results/competitor_gemini.json

    # Grok (via xAI OpenAI-compat endpoint)
    python scripts/eval_competitor.py \\
        --gt_jsonl  data/benchmark/ocrbench_v2.jsonl \\
        --api_base  https://api.x.ai/v1 \\
        --api_key   $XAI_API_KEY \\
        --model     grok-4 \\
        --out_json  results/competitor_grok.json
"""
import argparse
import base64
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

from scribe.eval.metrics import compute_all

# Published OCRBench V2 scores (higher is better — accuracy %)
OCRBENCH_V2_SOTA = {
    "Interfaze":         70.7,
    "Gemini-3.5-Flash":  63.9,
    "Claude-Sonnet-5":   59.2,
    "Gemini-3-Flash":    55.8,
    "Grok-4.3":          54.7,
    "GPT-5.4-Mini":      52.7,
}

_PROMPT = (
    "You are an expert OCR system. Extract all text from this document image exactly as it appears. "
    "Preserve layout, formatting, and all characters. Output only the extracted text, nothing else."
)


def _encode_image(image_path: str) -> tuple[str, str]:
    """Return (mime_type, base64_data)."""
    path = Path(image_path)
    ext = path.suffix.lower()
    mime = {"jpg": "image/jpeg", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return mime, data


def _call_api(
    image_path: str,
    api_base: str,
    api_key: str,
    model: str,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> str:
    mime, data = _encode_image(image_path)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ocrbench_v2_score(results: list[dict]) -> float:
    """OCRBench V2 scoring: exact match accuracy (higher is better)."""
    correct = sum(1 for r in results if r["pred"].strip() == r["gt"].strip())
    return round(100.0 * correct / max(len(results), 1), 2)


def _print_comparison(our_score: float, model_name: str, benchmark: str):
    print(f"\n{'─' * 55}")
    print(f"OCRBench V2 Comparison — {benchmark} vs Competitors")
    print(f"{'─' * 55}")
    print(f"{'Model':<25} {'Score':>8}  {'Status'}")
    print(f"{'-'*25} {'-'*8}  {'-'*12}")
    all_models = {model_name: our_score, **OCRBENCH_V2_SOTA}
    for name, score in sorted(all_models.items(), key=lambda x: -x[1]):
        tag = "← ours" if name == model_name else ("✓ better" if score > our_score else "✗ behind us")
        print(f"{name:<25} {score:>7.1f}%  {tag}")


def run(args):
    samples = [json.loads(l) for l in open(args.gt_jsonl, encoding="utf-8") if l.strip()]
    if args.limit:
        samples = samples[:args.limit]

    image_root = Path(args.image_root) if args.image_root else None
    model_label = args.model_label or args.model

    print(f"Competitor eval: {model_label}")
    print(f"Endpoint: {args.api_base}")
    print(f"Benchmark: {args.gt_jsonl} ({len(samples)} samples)\n")

    results = []
    errors = 0
    t0 = time.time()

    for sample in tqdm(samples, desc=model_label):
        image_path = str(image_root / sample["image"]) if image_root else sample["image"]
        try:
            pred = _call_api(
                image_path,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
            )
        except Exception as e:
            pred = ""
            errors += 1

        metrics = compute_all(pred, sample["text"])
        results.append({
            "image":    sample["image"],
            "category": sample.get("category", "all"),
            "pred":     pred,
            "gt":       sample["text"],
            "metrics":  metrics,
        })

        # Rate limiting
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)

    elapsed = time.time() - t0

    def avg(metric):
        return round(sum(r["metrics"][metric] for r in results) / max(len(results), 1), 4)

    overall = {m: avg(m) for m in ("ned", "cer", "wer", "bleu", "f1")}
    ocrbench_score = _ocrbench_v2_score(results)

    print(f"\nResults — {model_label}")
    print(f"  Samples  : {len(results)} ({errors} errors)")
    print(f"  Time     : {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/sample)")
    print(f"  NED↓     : {overall['ned']}")
    print(f"  CER↓     : {overall['cer']}")
    print(f"  F1↑      : {overall['f1']}")
    print(f"  OCRBench V2↑: {ocrbench_score}%")

    _print_comparison(ocrbench_score, model_label, args.gt_jsonl)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":        model_label,
            "api_base":     args.api_base,
            "benchmark":    args.gt_jsonl,
            "n_samples":    len(results),
            "n_errors":     errors,
            "elapsed_s":    round(elapsed, 2),
            "ocrbench_v2":  ocrbench_score,
            "overall":      overall,
            "samples":      results,
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved → {args.out_json}")


def main():
    p = argparse.ArgumentParser(description="Benchmark any OpenAI-compatible model on OCR")
    p.add_argument("--gt_jsonl",   required=True,  help="Ground truth JSONL (OCRBench V2 or OmniDocBench)")
    p.add_argument("--image_root", default="data/raw")
    p.add_argument("--api_base",   required=True,  help="OpenAI-compatible base URL")
    p.add_argument("--api_key",    required=True,  help="API key")
    p.add_argument("--model",      required=True,  help="Model name as accepted by the API")
    p.add_argument("--model_label", default=None,  help="Display name for tables (default: --model)")
    p.add_argument("--out_json",   default="results/competitor.json")
    p.add_argument("--limit",      type=int, default=0, help="Limit samples for quick test (0 = all)")
    p.add_argument("--timeout",    type=int, default=120, help="Per-request timeout in seconds")
    p.add_argument("--sleep_ms",   type=int, default=200, help="Sleep between requests (ms) for rate limiting")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
