# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

EnclaveScribe is a sovereign document intelligence and OCR system fine-tuned on Qwen2.5-VL-7B-Instruct. The goal is to outperform frontier model APIs (GPT, Claude, Gemini, Grok) and specialised cloud OCR (Interfaze.ai) on OCRBench V2 and OmniDocBench, while running entirely on-premise with zero per-query cost.

Primary benchmark target: **OCRBench V2 > 70.7%** (beat Interfaze.ai). Secondary: **OmniDocBench NED < 0.082**.

## Installation

```bash
# Core (inference + serving)
pip install -e .

# Training extras (GPU instances only)
pip install -e ".[train]"
# Note: Unsloth must be installed separately on GPU instances (CUDA-version-specific):
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Eval extras
pip install -e ".[eval]"

# Serving extras (choose one)
pip install -e ".[vllm]"
pip install -e ".[sglang]"
```

## Commands

### Data preparation
```bash
# Prepare all training datasets (~300K samples, ~150 GB, 4-8 hours)
bash scripts/prepare/run_all.sh
# Env vars: RAW_DIR, INTERIM_DIR, PROCESSED_DIR (defaults: data/raw, data/interim, data/processed)
```

### Training
```bash
# Single GPU (Unsloth auto-detected if available)
python scripts/train.py --config configs/train/lora.yaml

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train.py --config configs/train/lora_lambda_2xa100.yaml
```

### Inference (batch)
```bash
# Single file
python scripts/infer.py --file path/to/doc.pdf --model_dir outputs/lora --output_dir ./out

# Directory of documents
python scripts/infer.py --input_dir data/docs --model_dir outputs/lora --output_dir ./out
# Supported: images, PDFs, PPTs, DOCX. Output: markdown files.
```

### Serving (production)
```bash
# SGLang (~200 tok/sec on A100 80GB) — recommended for throughput
python scripts/serve.py --backend sglang --model outputs/lora

# vLLM (~150 tok/sec on A100 80GB) — OpenAI-compatible
python scripts/serve.py --backend vllm --model outputs/lora

# Local (no separate server, single-thread)
python scripts/serve.py --backend local --model outputs/lora

# With YAML config
python scripts/serve.py --config configs/serve/sglang.yaml --model outputs/lora
# FastAPI listens on :8080; inference backend on :10000 (SGLang) or :8000 (vLLM)
```

### Evaluation
```bash
# OmniDocBench (primary, NED lower is better)
python scripts/eval.py \
    --gt_jsonl data/benchmark/omnidocbench.jsonl \
    --image_root data/raw \
    --model_dir outputs/lora \
    --out_json results/omnidocbench.json

# OCRBench V2 competitor comparison
python scripts/eval_competitor.py --benchmark ocrbench_v2
```

### Tests
```bash
pip install pytest
pytest tests/
# Run a single test file
pytest tests/test_infer.py
```

## Architecture

### Package layout

```
scribe/          — core library (importable)
  model/         — Qwen2VLModel wrapper (load, infer, infer_batch)
  train/         — ScribeTrainer (HF Transformers) + Unsloth trainer + LoRA helpers
  infer/         — local.py (direct model call), vllm_client.py, sglang_client.py
  serving/       — subprocess launchers for vLLM and SGLang backends
  data/          — DocumentDataset / UnslothDocumentDataset (JSONL → PIL Image)
                   convert.py (PDF/PPT/DOCX → page images), collator, transforms
  postprocess/   — clean_output(): strips <|det|> region markers from model output
  eval/          — metrics.py: NED, CER, WER, BLEU-4, Token F1
  utils/         — logging

api/             — FastAPI gateway (routes/ocr.py: POST /v1/ocr)
scripts/         — CLI entry points (train.py, infer.py, eval.py, serve.py)
                   prepare/ — per-dataset prep scripts + run_all.sh
configs/         — YAML configs for model, train, infer, serve
tests/           — pytest unit tests
```

### Data flow

**Training:** JSONL files (`{"image": "path.png", "text": "ground truth"}`) → `DocumentDataset` / `UnslothDocumentDataset` → `DocumentCollator` → `ScribeTrainer` or Unsloth's `SFTTrainer`. Loss is computed only on the assistant response tokens (prompt tokens are masked to -100).

**Inference:** document file → `data/convert.py:to_images()` (renders each page at 300 DPI) → `Qwen2VLModel.infer()` → `postprocess/parser.py:clean_output()` → markdown text.

**Serving:** `scripts/serve.py` launches the chosen inference backend subprocess (vLLM or SGLang), waits for it to become healthy, then starts the FastAPI gateway. The gateway reads `SCRIBE_BACKEND`, `SCRIBE_SERVER_URL`, and `MODEL_PATH` env vars to route requests.

### Training backends

Two parallel training paths:
- **Unsloth** (preferred on GPU instances): ~2x faster, 60% less VRAM at full bf16. Uses `scribe/train/unsloth_trainer.py`. Auto-detected at runtime; falls back to HF if unavailable.
- **HuggingFace Transformers**: `ScribeTrainer` subclasses `Trainer` with custom loss masking. LoRA applied via `scribe/train/lora.py`.

### Config structure (YAML)

All train configs share the same schema:
- `model.path` — HF hub name or local checkpoint
- `lora.*` — LoRA rank, alpha, dropout, target_modules
- `dataset.train_jsonl` / `dataset.val_jsonl` / `dataset.image_root`
- `training_args.*` — passed directly to `TrainingArguments`
- `use_unsloth: true/false` — Unsloth override

### API

Single endpoint: `POST /v1/ocr` — accepts `{"image": "<base64>", "prompt": "document parsing.", "clean_output": true}`. Returns `{"text": "...", "tokens": N}`. Backend routing is done at startup via env vars, not per-request.
