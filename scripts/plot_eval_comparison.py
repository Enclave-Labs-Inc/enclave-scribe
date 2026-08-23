"""Bar-chart comparison of two eval runs (e.g. fine-tuned vs base model).

Input: two eval JSON files produced by `scripts/eval.py`.
Output: single PNG chart suitable for social media / reports.

Usage:
    python scripts/plot_eval_comparison.py \
        --a results/iter1_val_base.json      --a_label "Base" \
        --b results/iter1_val_finetuned.json --b_label "Iter-1" \
        --out reports/iter1/eval_comparison.png
"""
import argparse
import json
from pathlib import Path


def main():
    import matplotlib.pyplot as plt
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--a",       required=True, help="First eval JSON (e.g. baseline)")
    parser.add_argument("--b",       required=True, help="Second eval JSON (e.g. fine-tuned)")
    parser.add_argument("--a_label", default="A")
    parser.add_argument("--b_label", default="B")
    parser.add_argument("--out",     default="reports/iter1/eval_comparison.png")
    args = parser.parse_args()

    a = json.load(open(args.a))["overall"]
    b = json.load(open(args.b))["overall"]

    # Two groups: error metrics (lower is better) and quality metrics (higher is better).
    err_metrics = [("CER", "cer"), ("WER", "wer")]
    qual_metrics = [("BLEU", "bleu"), ("F1", "f1")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(2)
    w = 0.35

    a_err = [a[k] for _, k in err_metrics]
    b_err = [b[k] for _, k in err_metrics]
    ax1.bar(x - w/2, a_err, w, label=args.a_label, color="#9e9e9e")
    ax1.bar(x + w/2, b_err, w, label=args.b_label, color="#c832c8")
    ax1.set_xticks(x)
    ax1.set_xticklabels([m for m, _ in err_metrics])
    ax1.set_title("Error metrics (lower is better)", fontsize=13, weight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for i, (av, bv) in enumerate(zip(a_err, b_err)):
        ax1.text(i - w/2, av, f"{av:.2f}", ha="center", va="bottom", fontsize=10)
        ax1.text(i + w/2, bv, f"{bv:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")

    a_q = [a[k] for _, k in qual_metrics]
    b_q = [b[k] for _, k in qual_metrics]
    ax2.bar(x - w/2, a_q, w, label=args.a_label, color="#9e9e9e")
    ax2.bar(x + w/2, b_q, w, label=args.b_label, color="#c832c8")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m for m, _ in qual_metrics])
    ax2.set_ylim(0, 1)
    ax2.set_title("Quality metrics (higher is better)", fontsize=13, weight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for i, (av, bv) in enumerate(zip(a_q, b_q)):
        ax2.text(i - w/2, av, f"{av:.2f}", ha="center", va="bottom", fontsize=10)
        ax2.text(i + w/2, bv, f"{bv:.2f}", ha="center", va="bottom", fontsize=10, weight="bold")

    fig.suptitle(f"EnclaveScribe — {args.b_label} vs {args.a_label} on 23-sample val",
                 fontsize=15, weight="bold")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
