# EnclaveScribe

**A sovereign, self-hostable OCR model — Indic-first, deterministic document parsing.**

Built by [Enclave Labs](https://github.com/Enclave-Labs-Inc). MIT-licensed. See [VISION.md](VISION.md) for the full strategy and benchmark targets.

---

## Iteration 3 status — Devanagari OCR shipped

**LoRA fine-tune of `allenai/olmOCR-2-7B-1025` on 28,824 real Devanagari samples (himalaya-ai dataset). 8.7 hrs on g5.xlarge, ~$12 total. Held-out CER dropped from 1626% → 17.4% vs the base model on Devanagari.**

| Metric | Base OLMoCR-2-7B | Iter-3 (LoRA r=32) | Improvement |
|---|---:|---:|---:|
| CER ↓  | 16.26 (1626%) | **0.174 (17.4%)** | **~93×** |
| WER ↓  | 22.64 | **0.468** | 48× |
| F1 ↑   | 0.013 | **0.534** | 41× |
| Latency | 1.75 s/sample | **0.84 s/sample** | 2× faster |

Base OLMoCR-2 cannot read Devanagari at all — it hallucinates verbose English descriptions instead of transcribing, which is also why it's slower. Iter-3 is production-viable for single-word Devanagari OCR.

**Runbook**: [`reports/iter3_runbook.md`](reports/iter3_runbook.md)
**Artifacts**: `s3://enclave-scribe-checkpoints/outputs/iter3/` (380 MB LoRA adapter) · `s3://enclave-scribe-checkpoints/results/iter3/` (evals)

**Known defects** (fixable in iter-4, not adapter-specific):
- Generation loops on long dense pages (Qwen2.5-VL degeneration → `<tool_call>` token spam when it runs out of ideas)
- Word-level training data means the model wasn't optimized for full-page structure

**Iter-4 scope**: add page-level Devanagari data + generation-config fixes (`repetition_penalty`, stop-token handling).

---

## Iteration 1 (historical) — Latin OCR proof

Trained on $1.40 of AWS spot compute. Val CER dropped from 6.34 → 0.19 on CORD + FUNSD.

| Metric | Base Qwen2.5-VL-7B | Iter-1 (LoRA r=32) |
|---|---:|---:|
| CER ↓ | 6.34 | **0.19** |
| WER ↓ | 7.37 | **0.24** |
| BLEU ↑ | 0.00 | **0.65** |
| F1 ↑ | 0.15 | **0.88** |

Full report: [`reports/iter1/README.md`](reports/iter1/README.md). Val was a 2% random split from the same distribution as train, so most of the iter-1 gain reflects learning the output format — iter-3's numbers above are on a held-out benchmark from an entirely different distribution.

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

- **Iter-1 ✅** — 1,174 CORD + FUNSD samples, LoRA r=32, pipeline validated end-to-end.
- **Iter-2 ✅** — 30k mixed English OCR (DocVQA, XFUND, TextOCR, OmniDocBench, IDL), held-out benchmark, prompt-per-sample. Reference eval JSONs archived in `s3://enclave-scribe-checkpoints/results/iter2/`.
- **Iter-3 ✅** — 28,824 Devanagari samples on OLMoCR-2-7B, 93× CER improvement (see above).
- **Iter-4 (planned)**:
  1. Page-level Devanagari data (iter-3 was word-crops → weak on long-context structure)
  2. Fix generation-config: `repetition_penalty=1.1`, stop-token handling, better `max_new_tokens`
  3. Keep English performance from regressing (iter-3 gazette test showed English preserved, but no held-out English benchmark run against iter-3 yet)

See [VISION.md](VISION.md) for the long-term benchmark targets (OCRBench V2 > 70.7%, OmniDocBench NED < 0.082).

---

## License

MIT
