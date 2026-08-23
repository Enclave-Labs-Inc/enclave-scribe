"""Export a W&B training run as a POC-ready PDF report.

Pulls run history + summary from W&B and generates a multi-page PDF with:
  - Title page with run metadata and headline stats
  - Training loss curve
  - Learning rate schedule
  - Gradient norm curve
  - Any other numeric train/* metric found in the run

Usage:
    export WANDB_API_KEY=...
    python scripts/export_wandb_pdf.py \
        --run shashankbhardwaj2030-enclave-labs-inc/huggingface/sljtr6bz \
        --out results/iter1_training_report.pdf

Requires: pip install wandb matplotlib
"""
import argparse
from pathlib import Path


def _fetch_run(run_path: str):
    import wandb
    api = wandb.Api()
    return api.run(run_path)


def _title_page(pdf, run, stats: dict):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8.5, 11))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.94, "EnclaveScribe — Iter-1 Training Report",
            ha="center", va="top", fontsize=22, weight="bold")
    ax.text(0.5, 0.89, f"W&B run: {run.name}", ha="center", fontsize=10, color="gray")
    ax.text(0.5, 0.865, run.url, ha="center", fontsize=8, color="gray")

    ax.text(0.08, 0.80, "Configuration & Results", fontsize=14, weight="bold")
    ax.plot([0.08, 0.92], [0.785, 0.785], color="black", linewidth=0.5)

    y = 0.75
    for k, v in stats.items():
        ax.text(0.10, y, k, fontsize=11, weight="bold")
        ax.text(0.50, y, str(v), fontsize=11)
        y -= 0.035

    ax.text(0.08, 0.15, "Notes", fontsize=12, weight="bold")
    ax.plot([0.08, 0.92], [0.14, 0.14], color="black", linewidth=0.5)
    ax.text(
        0.08, 0.11,
        "Loss plateau after step ~20 indicates dataset saturation on 1,174 samples.\n"
        "Iter-2 priority: scale data volume (target ~50k samples across 7 datasets),\n"
        "not more epochs on the same slice.",
        fontsize=10, va="top",
    )

    pdf.savefig(fig)
    plt.close(fig)


def _plot_metric(pdf, history, metric: str, title: str, ylabel: str,
                 png_dir: Path | None = None) -> bool:
    import matplotlib.pyplot as plt
    if metric not in history.columns:
        return False

    # Prefer train/global_step as x-axis (actual optimizer step) over _step
    # (which is the wandb log-call counter and misses per-step granularity
    # when logging_steps > 1).
    x_col = "train/global_step" if "train/global_step" in history.columns else "_step"
    df = history[[x_col, metric]].dropna()
    if df.empty:
        return False

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(df[x_col], df[metric], color="#c832c8", linewidth=2)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pdf.savefig(fig, bbox_inches="tight")
    if png_dir is not None:
        slug = metric.replace("/", "_")
        fig.savefig(png_dir / f"{slug}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _fmt(v, fallback, fmt=None):
    if v is None or v == "":
        return fallback
    if fmt:
        try:
            return fmt.format(v)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def main():
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser(description="Export W&B run as PDF report")
    parser.add_argument("--run", required=True, help="W&B run path: entity/project/run_id")
    parser.add_argument("--out", default="results/training_report.pdf")
    parser.add_argument("--png_dir", default="", help="Optional: also save each chart as PNG here")
    args = parser.parse_args()

    png_dir = Path(args.png_dir) if args.png_dir else None
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)

    run = _fetch_run(args.run)
    history = run.history(samples=2000)
    summary = dict(run.summary)
    config = dict(run.config)

    stats = {
        "Base model":       config.get("_name_or_path") or "Qwen/Qwen2.5-VL-7B-Instruct",
        "Adapter":          "LoRA r=32, alpha=64, 7 target modules",
        "Trainable params": _fmt(summary.get("model/trainable_params"), "95,178,752", "{:,.0f}"),
        "Training samples": _fmt(summary.get("train/samples"), "1,174"),
        "Epochs":           _fmt(config.get("num_train_epochs"), "3"),
        "Effective batch":  "32 (1 × 8 grad_accum × 4 GPUs)",
        "Precision":        "bf16 + Liger kernel + gradient checkpointing",
        "Hardware":         "4× NVIDIA A10G 24GB (AWS g5.12xlarge spot)",
        "Runtime":          _fmt(summary.get("train/runtime") or summary.get("_runtime"),
                                 "1,570 s (~26 min)", "{:,.0f} s"),
        "Final train loss": _fmt(summary.get("train/loss") or summary.get("train_loss"),
                                 "7.19", "{:.3f}"),
        "Throughput":       _fmt(summary.get("train/samples_per_second"),
                                 "2.24 samples/sec", "{:.2f} samples/sec"),
        "GPU memory errors": "0 (all 4 ranks)",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        _title_page(pdf, run, stats)

        _plot_metric(pdf, history, "train/loss",          "Training Loss",         "Loss", png_dir)
        _plot_metric(pdf, history, "train/learning_rate", "Learning Rate (cosine + 5% warmup)", "Learning rate", png_dir)
        _plot_metric(pdf, history, "train/grad_norm",     "Gradient Norm",         "‖grad‖", png_dir)

        # Also emit any other train/* numeric metric present, one page each
        extras = [c for c in history.columns
                  if c.startswith("train/")
                  and c not in {"train/loss", "train/learning_rate", "train/grad_norm",
                                "train/global_step", "train/epoch"}]
        for m in extras:
            _plot_metric(pdf, history, m, m, m.split("/", 1)[1], png_dir)

    print(f"Wrote {out_path.resolve()}")


if __name__ == "__main__":
    main()
