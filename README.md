# EnclaveScribe

**A sovereign, self-hostable OCR model — Indic-first, deterministic document parsing.**

Built by [Enclave Labs](https://github.com/Enclave-Labs-Inc). MIT-licensed. See [VISION.md](VISION.md) for the full strategy and benchmark targets.

---

## Iteration 9 status — POSTMORTEM shipped, no HF publish, iter-10 direction locked

**Iter-9 aimed to push OCRBench V2 F1 from iter-7a's 0.499 → ≥ 0.60 (85% of Interfaze's 0.707). Missed hard.** Warm-started from iter-7a and trained on a docvqa-heavy 41k corpus. Both English ship gates FAILED. Devanagari gate PASSED thanks to a 500-sample replay.

| Benchmark | iter-7a | **iter-9** | Ship gate | Verdict |
|---|---:|---:|---:|---|
| OCRBench V2 F1 (300) | 0.499 | **0.484** (−3%) | ≥ 0.60 | ❌ **HARD FAIL** by 0.116 |
| OmniDocBench F1 (250) | 0.293 | **0.250** (−15%) | ≥ 0.291 | ❌ **HARD FAIL** by 0.041 |
| himalaya_500 CER ↓ | 4.463 | **0.442** (10× better) | ≤ 4.463 | ✅ PASS |

**Root cause — data mix, not the recipe.** The plan called for a 120k VQA-heavy corpus (OCR-VQA + ChartQA + TextVQA + InfographicVQA + DocVQA-extra). HF hub rate-limited every new source; only DocVQA + XFUND + IDL survived. After dropping IDL for time-budget reasons, the trained corpus was 87% DocVQA — the same distribution iter-7a already saturated on. Warm-starting on same-distribution data produced modest drift, not lift.

**Devanagari 10× improvement is the one bright spot.** 500 himalaya word crops as replay (1.2% of corpus) took CER from 446% (iter-7a) to 44% (iter-9). Confirms iter-8's diagnosis that iter-7a's Devanagari regression was underexposure, not architectural — and cheap to fix in any future iter.

**No adapter published to HF.** Iter-9 adapter kept on S3 at `s3://enclave-scribe-checkpoints/adapters/iter9/` for iter-10 diagnostic use. `enclavelabs/olmocr-2-iter7a-vqa` (iter-7a) remains the English VQA recommendation. `enclavelabs/enclave-scribe-devanagari` (iter-3) remains the Devanagari recommendation.

**Iter-10 direction (locked):** *rebuild the VQA corpus properly with a rate-limit-tolerant download strategy, then train fresh LoRA on base olmOCR-7B* (not warm-start). Pre-mirror OCR-VQA/ChartQA/TextVQA/InfographicVQA to S3 once, cap IDL at 20k, keep 1-2k himalaya replay. Ship gate relaxed to OCRBench V2 F1 ≥ 0.55. Cost budget $50-60.

- **📊 Full postmortem:** [`reports/iter9/POSTMORTEM.md`](reports/iter9/POSTMORTEM.md)
- **📁 Per-sample eval JSONs:** `s3://enclave-scribe-checkpoints/results/iter9/` (3 files)
- **📦 Adapter (S3-only, not on HF):** `s3://enclave-scribe-checkpoints/adapters/iter9/`
- **🔧 pip freeze:** [`reports/iter9/pip_freeze.txt`](reports/iter9/pip_freeze.txt)
- **⚙️ Config as-shipped:** [`configs/train/iter9_vqa.yaml`](configs/train/iter9_vqa.yaml)

**Cost: ~$42** (25.9h on g5.4xlarge on-demand us-east-1c). Under the $60 hard ceiling. Staging `i-073a0fe419ceb9f49` untouched throughout.

---

## Iteration 8 status — corrected measurements shipped, iter-7a reframed as VQA specialist

**Iter-8 fixed two long-standing eval bugs and re-measured all four models.** The results reshuffle everything we thought we knew about iter-7a.

**Headline: iter-7a is at 70.6% of Interfaze on OCRBench V2** (0.499 F1 vs Interfaze 0.707 target). Iter-6/7a's original 0.056 F1 was hidden by an eval bug — `scripts/eval.py` hardcoded `"document parsing."` as the prompt and silently dropped OCRBench V2's per-sample VQA questions. Fixed prompt routing (PR #62) exposed iter-7a's real strength: it's Enclave's best short-answer/VQA model by a wide margin.

| Benchmark (corrected) | base | iter-3 | iter-4 | **iter-7a** | Interfaze target |
|---|---:|---:|---:|---:|---:|
| OCRBench V2 F1 (300) | 0.093 | 0.099 | 0.115 | **0.499** | 0.707 |
| OmniDocBench F1 (250) | 0.291 | 0.016 | 0.106 | **0.293** | — |
| himalaya_500 CER (500 Devanagari word crops) | 14.32 | **0.175** | 0.231 | 4.463 | ≤ iter-4's 0.281 |

**Ship gate 2 (himalaya CER ≤ 28.1%) FAILED 15.9× for iter-7a.** The adapter catastrophically regressed Devanagari — DocVQA-heavy training killed most of what iter-3/iter-4 taught the model about Devanagari script.

**Strategic reframe — "choose your adapter" not "iter-7a replaces iter-4":**

| Use case | Recommended adapter |
|---|---|
| Devanagari word/page OCR | iter-3 or iter-4 |
| English document VQA / short-answer | **iter-7a** |
| English long-form page OCR | base olmOCR-7B (Enclave has nothing better yet) |

Iter-7a shipped to HF as a scoped VQA specialist ([`enclavelabs/olmocr-2-iter7a-vqa`](https://huggingface.co/enclavelabs/olmocr-2-iter7a-vqa)) with an explicit warning about Devanagari.

## Available models

Publicly available on HuggingFace:

| Model | Best for | Do NOT use for | Measured |
|---|---|---|---:|
| [`enclavelabs/enclave-scribe-devanagari`](https://huggingface.co/enclavelabs/enclave-scribe-devanagari) (iter-3) | Devanagari word/page OCR | non-Devanagari OCR | himalaya_500 CER **0.175** |
| [`enclavelabs/olmocr-2-iter7a-vqa`](https://huggingface.co/enclavelabs/olmocr-2-iter7a-vqa) (iter-7a) | English document VQA / short-answer | Devanagari (regressed 19× vs iter-4), long-form page OCR | OCRBench V2 F1 **0.499** |

Both adapters are LoRA on top of [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025) (Qwen2.5-VL-7B). MIT-licensed, fully sovereign, no API required at inference.

- **📊 Full writeup:** [`reports/iter8/CORRECTED_MEASUREMENTS.md`](reports/iter8/CORRECTED_MEASUREMENTS.md)
- **📁 Per-sample eval JSONs:** `s3://enclave-scribe-checkpoints/results/iter8/` (9 files)
- **📁 Devanagari benchmark (permanent):** `s3://enclave-scribe-checkpoints/data/benchmark/himalaya_500/` — 500 images, staged once, cheap to eval every iter now
- **🔧 pip freeze:** [`reports/iter8/pip_freeze.txt`](reports/iter8/pip_freeze.txt)

---

## Iteration 7a status — multilingual expansion shipped

**First fine-tune that doesn't regress base olmOCR-7B on English.** Fresh LoRA r=32/α=64 on `allenai/olmOCR-2-7B-1025`, ~9.8k filtered multilingual samples (DocVQA + XFUND + IDL-WDS), 2 epochs, ~$48 on g5.4xlarge. **Iter-8's corrected measurements above supersede the numbers originally reported here.**

| OmniDocBench (250 pages, unchanged in iter-8) | NED ↓ | BLEU ↑ | F1 ↑ |
|---|---:|---:|---:|
| iter-3 | 1.137 | 0.012 | 0.016 |
| iter-4 | 0.983 | 0.057 | 0.106 |
| base olmOCR-2-7B-1025 | 2.998 | 0.131 | 0.291 |
| **iter-7a** | **1.955** | **0.142** | **0.293** |
| Interfaze target | 0.082 | — | — |

**Original read (superseded):** "matches base F1 (+0.7%, inside noise); training loss stayed flat at ~6.2 because 87% of corpus was DocVQA." **Iter-8 reframe:** matching base on OmniDocBench was a red herring. DocVQA training produced a specialist VQA model whose real value hid behind OCRBench V2 (0.499 F1, 5.4× base) — invisible until the per-sample-prompt eval bug was fixed.

- **📊 Iter-7a original writeup:** [`reports/iter7a/README.md`](reports/iter7a/README.md)
- **📊 Corrected numbers:** [`reports/iter8/CORRECTED_MEASUREMENTS.md`](reports/iter8/CORRECTED_MEASUREMENTS.md)
- **📦 Adapter (S3):** `s3://enclave-scribe-checkpoints/adapters/iter7a/`
- **⚙️ Config**: [`configs/train/iter7a_multilingual.yaml`](configs/train/iter7a_multilingual.yaml)

---

## Iteration 6 status — baseline measurement shipped, iter-7 = multilingual expansion

**First-ever measurement of iter-3, iter-4, and base olmOCR-7B against VISION.md's actual benchmarks:**

| OmniDocBench (250 pages) | NED ↓ | BLEU ↑ | F1 ↑ |
|---|---:|---:|---:|
| iter-3 (Devanagari word LoRA) | 1.137 | 0.012 | 0.016 |
| iter-4 (Devanagari page LoRA) | 0.983 | 0.057 | 0.106 |
| **base olmOCR-7B** (no LoRA) | 2.998 | **0.131** | **0.291** |
| Interfaze target | 0.082 | — | — |

**Key finding:** base olmOCR-7B has **2.7× better F1 than iter-4** on English page content. Our Devanagari-focused fine-tuning has actively degraded English/multilingual capability — the very benchmarks the vision cares about.

**Iter-7 direction (decided by iter-6):** Branch 7A — expand training data toward VISION.md's 300k multilingual target. Prep scripts for DocVQA/TextOCR/HierText/XFUND/IDL-WDS already committed. Ship gate: F1 on OmniDocBench ≥ base olmOCR-7B's 0.291 while preserving Devanagari word CER ≤ iter-4's 28.1%.

- **📊 Full measurement report:** [`reports/iter6/BASELINE_MEASUREMENT.md`](reports/iter6/BASELINE_MEASUREMENT.md)
- **📁 Benchmark JSONLs:** `s3://enclave-scribe-checkpoints/data/benchmark/omnidocbench_test.jsonl` (1,645 samples) + `.../ocrbench_v2.jsonl` (10k samples)
- **📁 Per-sample eval outputs:** `s3://enclave-scribe-checkpoints/results/iter6/` (6 JSONs)

---

## Iteration 5 status — postmortem shipped, no adapter, iter-6 elevated

**Iter-5 hit three independent walls and no adapter was trained. Full postmortem: [`reports/iter5/POSTMORTEM.md`](reports/iter5/POSTMORTEM.md).**

Short version:
1. **`allenai/olmOCR-2-32B-1025` doesn't exist.** AllenAI never published a 32B olmOCR. The 2026-09-10 plan referenced a phantom repo.
2. **32B + FSDP + LoRA does not fit 4× A10G 24GB.** Substituting `Qwen/Qwen2.5-VL-32B-Instruct` on `g5.12xlarge` produced 6 crashes in 90 min (GPU OOM → CPU OOM). Design preserved at [`configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml`](configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml) for when P/g5.48xlarge quota lands.
3. **7B fine-tuning environment drift.** On the *exact hardware iter-4 shipped on* (g5.4xlarge, 1× A10G), we couldn't reproduce iter-4's setup: `transformers==4.55.4` has a memory regression (OOM at 20 GB even at iter-4's exact hyperparameters), and every earlier version we tried has a Qwen2.5-VL-specific bug. Iter-4's exact `pip freeze` was never recorded — that's now a non-negotiable for iter-6.

The still-open strategic question — is iter-4's ceiling **corpus quality** or **model capacity**? — remains unresolved. In the absence of evidence, iter-6 treats pseudo-labels as the working ceiling and moves to human labels.

Pre-iter-5 dry-run measurements (still the load-bearing baseline for iter-6's ship gate):

| Metric (Devanagari word, 500 samples) | iter-3 | iter-4 | Δ |
|---|---:|---:|---:|
| CER ↓ | **0.2151** | **0.2808** | +0.0657 ❌ regression |
| F1  ↑ | 0.5609 | **0.5976** | +0.0367 ✅ better |

- **📝 Postmortem**: [`reports/iter5/POSTMORTEM.md`](reports/iter5/POSTMORTEM.md)
- **📝 Original dry-run writeup**: [`reports/iter5/DRYRUN.md`](reports/iter5/DRYRUN.md)
- **⚙️ Retired 32B config** (for future re-run when quota lands): [`configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml`](configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml)
- **🌱 Iter-6 human-labels pilot (now critical path)**: [`reports/iter6/PILOT.md`](reports/iter6/PILOT.md) · review tool at [`scripts/label/review.py`](scripts/label/review.py)

## Iteration 4 status — Page-level Devanagari shipped

**LoRA fine-tune resuming from iter-3, on 793 pseudo-labeled Devanagari page images (ai4bharat/indicdlp) + 500 word replay samples. 1h37m on g5.xlarge, ~$3 training. Fixes iter-3's dead-loop failure mode on long dense pages.**

Ship-gate: 6-page Gazette of India Extraordinary notification with the runtime `bad_words_ids` workaround **DISABLED** in `extract_page`:

| Signal | Iter-3 | **Iter-4** |
|---|---:|---:|
| Total chars extracted | 7,122 | **13,646** |
| Pages dead-looped | 2/6 | **0/6** |
| Wallclock (6 pages) | 20 min | **12 min** |

Iter-3 dead-looped on pages 4 and 5 (0 chars output, `<tool_call>` blocks stripped by post-hoc regex). Iter-4 extracts every page cleanly with no loops.

Iter-4 is also more source-faithful — for example, it preserves Arabic numerals inside the English section where iter-3 hallucinates Devanagari digits (`२११३` → `2113`).

- **📦 Model on HuggingFace**: [enclavelabs/enclave-scribe-devanagari-iter4](https://huggingface.co/enclavelabs/enclave-scribe-devanagari-iter4) (iter-3 remains at [enclavelabs/enclave-scribe-devanagari](https://huggingface.co/enclavelabs/enclave-scribe-devanagari))
- **📝 Full writeup**: [`reports/iter4/README.md`](reports/iter4/README.md)
- **🔬 Head-to-head vs iter-3**: [`reports/iter4/GAZETTE_TEST.md`](reports/iter4/GAZETTE_TEST.md)
- **🧪 Canonical failure case**: [`tests/fixtures/pdfs/`](tests/fixtures/pdfs/)
- **🗂️ Artifacts**: `s3://enclave-scribe-checkpoints/outputs/iter4/` (adapter) · `s3://enclave-scribe-checkpoints/results/iter4/` (ship-gate)

### Honest read

Training loss stayed flat at ~4.2 across all 82 steps (grad norms 0.2–0.9). Iter-4's win is a class-of-failure fix (no more dead-looping) rather than a broad character-level uplift. It's "iter-3 that doesn't break on long pages", not "iter-3 but sharper". Iter-5 will pursue either real human labels or a bigger base model.

The runtime `bad_words_ids` workaround stays in production as belt-and-suspenders.

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

- **📦 Model on HuggingFace**: [enclavelabs/enclave-scribe-devanagari](https://huggingface.co/enclavelabs/enclave-scribe-devanagari)
- **📝 Full writeup**: [`reports/iter3/README.md`](reports/iter3/README.md)
- **▶️ Runbook**: [`reports/iter3_runbook.md`](reports/iter3_runbook.md)
- **🗂️ Artifacts**: `s3://enclave-scribe-checkpoints/outputs/iter3/` (adapter) · `s3://enclave-scribe-checkpoints/results/iter3/` (evals)

### Use it in 10 lines

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

processor = AutoProcessor.from_pretrained("allenai/olmOCR-2-7B-1025")
model = AutoModelForImageTextToText.from_pretrained(
    "allenai/olmOCR-2-7B-1025", dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "enclavelabs/enclave-scribe-devanagari")
# ...then generate with repetition_penalty=1.1 — see the HF model card for the full snippet.
```

**Known defects** (fixable in iter-4, not adapter-specific):
- Generation loops on long dense pages (Qwen2.5-VL emits `<tool_call>` token spam when it runs out of ideas). Worked around in the agent via `repetition_penalty=1.1` + `bad_words_ids`.
- Word-level training data means the raw model isn't optimized for full-page structure — for pages use the agent, not the raw adapter.

**Iter-4 scope**: page-level Devanagari data + formal English-regression benchmark.

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
- **Iter-4 ✅** — Hybrid bootstrap: 793 pseudo-labeled IndicDLP pages + 500 word replay, resumed from iter-3. Fixes dead-loop failure mode on long dense pages. See "Iteration 4" above.
- **Iter-5 📄 postmortem shipped, no adapter** — target model `allenai/olmOCR-2-32B-1025` doesn't exist on HF; 32B substitute doesn't fit 4× A10G; 7B fine-tune blocked by env drift on the exact hardware iter-4 shipped on. Retired 32B config preserved for a future run when P/g5.48xlarge quota lands. See [`reports/iter5/POSTMORTEM.md`](reports/iter5/POSTMORTEM.md).
- **Iter-6 📊 baseline measurement shipped** — first-ever measurement of our adapters against VISION.md's actual benchmarks (OmniDocBench + OCRBench V2). Finding: **base olmOCR-7B has 2.7× better F1 recall on English pages than iter-4** — our Devanagari-focused fine-tuning actively degraded English/multilingual capability. See [`reports/iter6/BASELINE_MEASUREMENT.md`](reports/iter6/BASELINE_MEASUREMENT.md).
- **Iter-7a ✅ shipped, matches base olmOCR-7B on English** — fresh LoRA r=32/α=64 on olmOCR-7B, 9.8k filtered multilingual corpus (DocVQA + XFUND + IDL-WDS after HierText/TextOCR URL rot forced drops). OmniDocBench F1 = 0.293 (vs base 0.291, +0.7%); OCRBench V2 F1 = 0.056 (2.2× base). NED 1.955 (35% shorter output than base). Ship gate 1 & 3 passed; gate 2 (Devanagari) not measured — 60 GB himalaya download deferred. Training loss stayed flat at ~6.2 because 87% of corpus was DocVQA short-answer pairs in pure-OCR framing — corpus design was the ceiling. Full postmortem: [`reports/iter7a/README.md`](reports/iter7a/README.md).
- **Iter-7a ✅ RESULT UPDATE (see iter-8):** OmniDocBench F1 numbers above are correct. OCRBench V2 F1 was actually **0.499** (not the 0.056 originally reported — eval bug). himalaya_500 CER = **4.463 (446%) — ship gate 2 hard-failed 15.9×**. Iter-7a reframed as VQA specialist, not general-OCR replacement.
- **Iter-8 ✅ eval-infrastructure shipped** — fixed the per-sample-prompt bug in `scripts/eval.py` (PR #62), staged the 500-image Devanagari benchmark on S3 permanently, re-measured all 4 adapters × 3 benchmarks. Reveals iter-7a at **70.6% of Interfaze's OCRBench V2 target** (0.499 / 0.707) — was invisible until eval fix. See [`reports/iter8/CORRECTED_MEASUREMENTS.md`](reports/iter8/CORRECTED_MEASUREMENTS.md).
- **Iter-9 🎯 = 9A (VQA specialization — push OCRBench V2 F1 from 0.50 to ≥ 0.60)** — double down on iter-7a's accidental VQA lead. Expanded VQA corpus (~120k: OCR-VQA + ChartQA + TextVQA + InfographicVQA + DocVQA-extra + iter-7a-baseline), **warm-started from iter-7a** (LR 5e-5, 1 epoch — extension not full retrain), ~$25-40. Ship gate: OCRBench V2 F1 ≥ 0.60 (85% of Interfaze) AND OmniDocBench F1 ≥ 0.291. Path B (Devanagari rehab) and Path C (real page-OCR corpus) documented in iter-8 report as fallbacks if iter-9A doesn't clear its gate.
- **Standing gate for all future iterations**: [`tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf`](tests/fixtures/pdfs/) as a canonical failure case; `data/benchmark/himalaya_500.jsonl` (on S3) as the Devanagari word-level regression gate.

See [VISION.md](VISION.md) for the long-term benchmark targets (OCRBench V2 > 70.7%, OmniDocBench NED < 0.082).

---

## License

MIT
