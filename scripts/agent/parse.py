"""CLI entry point for the agentic OCR pipeline.

Usage:
    python scripts/agent/parse.py \
        --pdf   /path/to/document.pdf \
        --out   output.md

    # With a fine-tuned adapter on top of the base VLM:
    python scripts/agent/parse.py \
        --pdf         /path/to/document.pdf \
        --out         output.md \
        --adapter_dir outputs/iter1

    # Higher DPI (bigger images, better small-text OCR, slower):
    python scripts/agent/parse.py --pdf doc.pdf --out out.md --dpi 300
"""
import argparse
import json
from pathlib import Path

from scribe.agent import Agent, ModelRegistry


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe agentic OCR")
    parser.add_argument("--pdf",         required=True,  help="Input PDF path")
    parser.add_argument("--out",         required=True,  help="Output markdown path")
    parser.add_argument("--dpi",         type=int, default=200, help="PDF render DPI (default 200)")
    parser.add_argument("--base_model",  default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_dir", default="",     help="Optional LoRA adapter")
    parser.add_argument("--report_json", default="",     help="Optional structured report path")
    args = parser.parse_args()

    registry = ModelRegistry(
        vlm_path=args.base_model,
        vlm_adapter_dir=args.adapter_dir,
    )
    agent = Agent(registry=registry, verbose=True)
    result = agent.parse(args.pdf, dpi=args.dpi)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.markdown, encoding="utf-8")
    print(f"\nWrote markdown → {out_path.resolve()}")

    if args.report_json:
        report = {
            "pdf":         result.pdf_path,
            "n_pages":     result.n_pages,
            "elapsed_s":   round(result.elapsed_s, 2),
            "validation":  result.validation,
            "per_page":    [
                {
                    "page":       p.page_index,
                    "n_regions":  p.n_regions,
                    "elapsed_s":  round(p.elapsed_s, 2),
                    "labels":     [lbl for lbl, _ in p.extractions],
                }
                for p in result.pages
            ],
        }
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2))
        print(f"Wrote report  → {Path(args.report_json).resolve()}")


if __name__ == "__main__":
    main()
