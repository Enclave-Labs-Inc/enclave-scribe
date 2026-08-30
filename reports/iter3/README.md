# EnclaveScribe — Iteration 3

**Fine-tuning OLMoCR-2-7B for Devanagari OCR on $12 of AWS compute.**

---

## TL;DR

| Metric | Base OLMoCR-2-7B | Iter-3 (LoRA r=32) | Change |
|---|---:|---:|---:|
| **CER** ↓ | 16.26 (1626%) | **0.174 (17.4%)** | ~93× better |
| **WER** ↓ | 22.64 | **0.468** | 48× better |
| **F1** ↑ | 0.013 | **0.534** | 41× better |
| **Latency** | 1.75 s/sample | **0.84 s/sample** | 2× faster |

500-sample held-out benchmark of single-word Devanagari OCR (himalaya-ai dataset).

Base OLMoCR-2 cannot read Devanagari — it hallucinates verbose English descriptions instead of transcribing (which is also why it's slower: more tokens generated). Iter-3 is production-viable for single-word Indic OCR.

**Honest caveat**: training data was word-level image crops, so the raw model plateaus on long dense pages. Iter-4 will add page-level Devanagari scans to close that gap. See "Known defects" below.

---

## What we built

- **Base model**: [allenai/olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025) — Qwen2.5-VL-7B fine-tuned by Allen AI for English document OCR
- **Adapter**: LoRA r=32, α=64, 380 MB safetensors
- **Data**: 28,824 word-level Devanagari samples from [himalaya-ai/devanagari_ocr_dataset](https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_dataset) (Hindi, Marathi, Sanskrit, Nepali, Pali — mix of printed and IIT-Indic-HW handwritten)
- **Hardware**: 1× NVIDIA A10G 24GB on AWS g5.xlarge on-demand
- **Runtime**: 8.7 hours (31,240 seconds), 901 steps, 1 epoch
- **Precision**: bf16 + Liger kernel + gradient checkpointing
- **Effective batch**: 32 (per-device batch 1 × grad accum 32 on single GPU)
- **LR**: 1.5e-4 cosine schedule, 27 warmup steps

## Training curve

Loss dropped from ~7.5 to ~6.78 and plateaued for the second half of training — same saturation pattern as iter-1 on CORD. Word-level data is small and repetitive; the model hits its ceiling early. Iter-4 needs richer data, not more epochs.

Final `train_loss: 6.903` (average), grad_norm stable around 0.08 throughout.

## Real-world qualitative test

Ran the fine-tuned model through the agent pipeline (`scripts/agent/parse.py`) on a 9-page Ministry of Environment gazette PDF (bilingual Hindi + English, dense tables, seals):

- **✅ Devanagari transcription** — clean Unicode Devanagari, bilingual layout preserved
- **✅ English preserved** — no regression on Latin script
- **✅ Tables emitted as `<table>` HTML** — rowspan/colspan handled correctly
- **⚠️ Long dense pages triggered generation loops** — see "Known defects"

## Known defects (and the post-hoc fix)

The raw model has two failure modes on long dense pages, both fixed via generation config (no retraining needed):

1. **`<tool_call>` token spam** — Qwen2.5-VL emits `<tool_call>...</tool_call>` blocks when it runs out of ideas. The tokenizer doesn't mark these special so `skip_special_tokens=True` doesn't strip them.
2. **Greedy decoding loops** — the greedy decoder gets stuck repeating short sequences until it hits `max_new_tokens=4096`.

**Fix** (see [PR #41](https://github.com/Enclave-Labs-Inc/enclave-scribe/pull/41)): add `repetition_penalty=1.1` + block the `<tool_call>` token via `bad_words_ids`. Verified on the same gazette PDF:

| Page | Before fix | After fix |
|------|-----------:|----------:|
| 6 | 12,673 chars (loop) | **3,413 chars** ✓ |
| 7 | 12,695 chars (loop) | **3,372 chars** ✓ |
| 8 | 13,333 chars (loop) | **2,738 chars** ✓ |
| `<tool_call>` count | many | **0** |

The generation fix is a workaround. The real fix is iter-4 with page-level training data.

## What broke along the way

1. **Dataset schema mismatch** — himalaya-ai advertised ShareGPT format but real schema is WebDataset tar.gz (`__key__` + `png`) plus a separate 2.15 GB `devanagari_ocr.json` keyed by image path. Prep script rewritten (PR #39) to download the JSON, build a `{stem → text}` lookup, and match samples by key.
2. **Python 3.11.0rc1 too old** — Ubuntu 22.04's `python3.11-venv` installed a release candidate predating `sys.get_int_max_str_digits`, breaking torch 2.13. Fix: use the AMI's `/opt/pytorch` venv (Python 3.12.10 + torch 2.7).
3. **`warmup_ratio` removed** — newer transformers rejects it. Patched config to `warmup_steps: 27`.
4. **`g5.12xlarge` InsufficientInstanceCapacity** — AWS blocked launches; dropped to g5.xlarge which was cheaper anyway (~$1/hr vs $6/hr).
5. **Spot quota 0** — used on-demand.
6. **`HF_TOKEN` not read from env** — `huggingface_hub` needs `hf auth login --token $HF_TOKEN` to persist to disk.

## Cost

| Item | Cost |
|---|---|
| Training compute (8.7 hrs × g5.xlarge on-demand) | ~$9 |
| Data prep + debug + eval on same instance | ~$3 |
| Generation-config fix test on fresh instance (~1 hr) | ~$1 |
| **Total for iter-3 (soup to nuts)** | **~$13** |

## What's next — iter-4

Iter-3 proved Devanagari OCR works at the word level. Iter-4 needs to make it work at the *page* level so we can drop the generation-config workaround.

**Priorities:**
1. **Page-level Devanagari data** — Nayana, IndicVisionBench have full-page Hindi scans. Target ~5-10k page images.
2. **Keep the generation-config fix as a belt-and-suspenders** — even with better training data, `repetition_penalty` is cheap insurance.
3. **English regression check** — iter-3 gazette test showed English preserved qualitatively, but we never ran a held-out English benchmark against iter-3. Iter-4 eval should include both.

---

## Links

- **Model on HuggingFace**: [enclavelabs/enclave-scribe-devanagari](https://huggingface.co/enclavelabs/enclave-scribe-devanagari)
- **Repo**: https://github.com/Enclave-Labs-Inc/enclave-scribe
- **Adapter checkpoint**: `s3://enclave-scribe-checkpoints/outputs/iter3/`
- **Eval results**: `s3://enclave-scribe-checkpoints/results/iter3/` (pre-genfix) and `results/iter3-genfix/` (post)

## Reproduce

```bash
# Prep 30k Devanagari samples (~1-2 hrs, ~50 GB disk)
python scripts/prepare/prep_himalaya_indic.py \
  --raw_dir data/raw \
  --out_jsonl data/interim/himalaya_indic.jsonl \
  --benchmark_jsonl data/benchmark/himalaya_indic_test.jsonl \
  --max_samples 30000

python scripts/prepare/merge.py \
  --interim_dir data/interim \
  --out_dir data/processed --val_ratio 0.02

# Train (~8.7 hrs on g5.xlarge)
python scripts/train.py --config configs/train/iter3.yaml

# Eval on held-out
python scripts/eval.py \
  --gt_jsonl data/benchmark/heldout_test.jsonl \
  --image_root data/raw \
  --base_model allenai/olmOCR-2-7B-1025 \
  --adapter_dir outputs/iter3 \
  --out_json results/iter3_heldout.json
```

Full runbook: [`reports/iter3_runbook.md`](../iter3_runbook.md)
