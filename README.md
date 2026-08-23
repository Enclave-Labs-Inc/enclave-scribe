# EnclaveScribe

**A sovereign, self-hostable document OCR model fine-tuned from Qwen2.5-VL-7B.**

Built by [Enclave Labs](https://github.com/Enclave-Labs-Inc). MIT-licensed. See [VISION.md](VISION.md) for the full strategy and benchmark targets.

---

## Iteration 1 status

**Trained on $1.40 of AWS spot compute. Val CER dropped from 6.34 → 0.19 vs base model.**

![Iter-1 vs base Qwen2.5-VL-7B on 23-sample val](reports/iter1/eval_comparison.png)

| Metric | Base Qwen2.5-VL-7B | Iter-1 (LoRA r=32) |
|---|---:|---:|
| CER ↓ | 6.34 | **0.19** |
| WER ↓ | 7.37 | **0.24** |
| BLEU ↑ | 0.00 | **0.65** |
| F1 ↑ | 0.15 | **0.88** |

**Full report**: [`reports/iter1/README.md`](reports/iter1/README.md)

**Caveat**: val is a 2% random split from the same distribution as train (CORD + FUNSD). Most of the gain reflects the model learning our output format — iter-2 will measure real generalization on a held-out test set.

---

## Quick start

### Install

```bash
git clone https://github.com/Enclave-Labs-Inc/enclave-scribe.git
cd enclave-scribe
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Prep data

```bash
bash scripts/prepare/run_all.sh
```

CORD and FUNSD produce samples out of the box; the other 5 prep scripts have known bugs being fixed in iter-2 (see roadmap below).

Official test splits (CORD, FUNSD) are automatically routed to `data/benchmark/heldout_test.jsonl` and kept out of the training pool — use this file for stable, apples-to-apples evaluation across iterations.

### Train (4× A10G, ~26 min, ~$1.40 spot)

```bash
torchrun --nproc_per_node=4 scripts/train.py \
  --config configs/train/unsloth_aws.yaml
```

Config sets `use_unsloth: false` — trains via HuggingFace Trainer + DDP across all 4 GPUs (Unsloth free tier is single-GPU only).

### Eval

```bash
# Fine-tuned model
python scripts/eval.py \
  --gt_jsonl    data/processed/val.jsonl \
  --image_root  data/raw \
  --base_model  Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter_dir outputs/iter1 \
  --out_json    results/iter1_val_finetuned.json

# Base model (baseline)
python scripts/eval.py \
  --gt_jsonl   data/processed/val.jsonl \
  --image_root data/raw \
  --base_model Qwen/Qwen2.5-VL-7B-Instruct \
  --out_json   results/iter1_val_base.json
```

See [`configs/eval/iter1.yaml`](configs/eval/iter1.yaml) for command variants.

### Generate a training report

```bash
export WANDB_API_KEY=<your-key>
python scripts/export_wandb_pdf.py \
  --run <entity>/<project>/<run_id> \
  --out results/training_report.pdf \
  --png_dir reports/iter1
```

---

## Repo layout

```
scribe/           Python package
├── data/         JSONL datasets, image collator, augmentations
├── train/        HF Trainer + Unsloth training paths, LoRA config
├── eval/         CER/WER/BLEU/F1 metrics
├── model/        Qwen2.5-VL loader (base + PEFT adapter)
├── infer/        Local + vLLM/SGLang inference
└── postprocess/  Output cleanup

configs/
├── train/        Training configs (YAML)
├── eval/         Eval configs
└── model/        Model configs

scripts/
├── prepare/      Per-dataset prep pipelines (CORD, FUNSD, XFUND, ...)
├── train.py      Training entry point (single- or multi-GPU via torchrun)
├── eval.py       Eval on any ground-truth JSONL
├── infer.py      Single-image inference CLI
├── export_wandb_pdf.py     W&B run → PDF/PNG report
└── plot_eval_comparison.py Bar-chart comparison of two eval runs

reports/          Per-iteration writeups with charts
```

---

## Roadmap

**Iter-1 ✅ (complete)** — 1,174 samples, LoRA r=32, 26 min training, pipeline validated end-to-end.

**Iter-2 (in progress)**:
1. Reserve official CORD/FUNSD test splits as a stable held-out benchmark (never in training)
2. Fix 3 broken prep pipelines: XFUND, TextOCR, OmniDocBench
3. Add DocVQA → target ~30k training samples across 5+ dataset types
4. Prompt-per-dataset — teach the model to switch output formats via prompt
5. LoRA r=64, 2 epochs
6. Real eval on held-out test + out-of-distribution documents

Estimated iter-2 compute: ~3-4 hours × g5.12xlarge spot ≈ $12–15.

See [VISION.md](VISION.md) for the long-term benchmark targets (OCRBench V2 > 70.7%, OmniDocBench NED < 0.082).

---

## License

MIT
