# Iter-6 baseline measurement — where do we actually stand?

**Date:** 2026-09-12
**Iteration outcome:** measurement report (no adapter trained this iteration, per plan)
**Compute cost:** ~$11 (g5.4xlarge on-demand, 6h 28m wallclock, us-east-1)
**Instance:** `i-08891251d4f7848ee` (terminated on completion). Staging `i-073a0fe419ceb9f49` untouched.

---

## Why this exists

`VISION.md` names two concrete competitive targets:

- **OCRBench V2 > 70.7%** (beat Interfaze.ai)
- **OmniDocBench NED < 0.082** (beat Unlimited-OCR)

Iter-1 through iter-5 shipped four adapters and never once measured against either benchmark. Every "success gate" we've used to date (Devanagari word CER on `himalaya_500.jsonl`, gazette char count, `<tool_call>` blocks) has been a proxy. Iter-5's postmortem made the diagnosis explicit: we were "choosing an experiment without first knowing where the actual gap is."

Iter-6 measures three adapters (iter-3, iter-4, base olmOCR-2-7B-1025) against both benchmarks to close that blind spot.

## Methodology

- **Benchmarks**: OmniDocBench (`opendatalab/OmniDocBench`, 1,651 pages) and OCRBench V2 (`lmms-lab/OCRBench-v2`, 10k samples).
- **Subsampling**: 250 OmniDocBench pages, 300 OCRBench V2 samples per adapter. The full sets were infeasible on 1× A10G at 123 s/sample (plan rollback #2 triggered — see the plan file's compute plan).
- **Token caps**: `max_new_tokens=768` for OmniDocBench (full-page markdown), `128` for OCRBench V2 (short-answer VQA). Even at 768, some outputs get truncated — that's a real ceiling.
- **Prompt**: fixed `"document parsing."` prompt via `scribe.infer.local.infer_image()`. **Important caveat for OCRBench V2**: that benchmark uses per-sample questions ("What is the mass shown in the image?", etc.). Our fixed prompt makes the model dump full OCR instead of answering — so absolute OCRBench V2 scores are not comparable to Interfaze's published 70.7%. Numbers are internally comparable across our own adapters.
- **Scoring**: `scripts/eval.py` computes NED, CER, WER, BLEU-4, token-F1 via `scribe/eval/metrics.py:compute_all()`.
- **Env**: DL AMI `/opt/pytorch` conda env (Python 3.11 + torch 2.4.1 + flash-attn 2.4.2) + `transformers==4.55.4`, `peft==0.20.0`, `accelerate==1.4.0`. `HF_HUB_DISABLE_XET=1` to dodge a rate-limit issue during OmniDocBench prep.

## Results — OmniDocBench (250 pages, mnt=768)

| Adapter | NED ↓ | CER ↓ | WER ↓ | BLEU ↑ | F1 ↑ | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| **iter-3** (Devanagari word LoRA, r=32) | 1.137 | 1.137 | 1.171 | 0.012 | 0.016 | 3h 07m |
| **iter-4** (Devanagari page LoRA, r=32 resumed) | 0.983 | 0.983 | 1.115 | 0.057 | 0.106 | 44m |
| **base olmOCR-2-7B-1025** (no LoRA) | 2.998 | 2.998 | 2.686 | **0.131** | **0.291** | 1h 32m |
| **VISION.md target** (Interfaze) | 0.082 | — | — | — | — | — |

**Key observation:** the base model has **2.7× better F1 than iter-4** on English page content. Our LoRA fine-tuning has actively degraded the model's English recall — it's forgotten most of the base's OCR competence in exchange for Devanagari word-level accuracy.

Why NED > 1 for all rows: NED is `edit_distance / len(ground_truth)`. When the prediction is *longer* than ground truth (excess text, hallucinated repetition, or long verbose completions), NED exceeds 1. Base's NED=2.998 with F1=0.291 tells us: base captures most of the right content (0.291 F1) but also emits ~3× the length in extra tokens. Our fine-tunes emit shorter outputs (lower NED) but at the cost of dropping actual content (lower F1).

**Interfaze's published 0.082 is likely computed with OmniDocBench's official layout-aware scorer** (which normalizes structure). Our simple text-diff NED is stricter. The takeaway is the *ordering* — base beats iter-4 beats iter-3 on English content recall — not the absolute number vs Interfaze.

## Results — OCRBench V2 (300 samples, mnt=128, fixed prompt)

| Adapter | NED ↓ | CER ↓ | WER ↓ | BLEU ↑ | F1 ↑ | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| **iter-3** | 36.5 | 36.5 | 22.6 | 0.001 | 0.026 | 32m |
| **iter-4** | 10.5 | 10.5 | 6.9 | 0.000 | 0.006 | 7m |
| **base olmOCR-2-7B-1025** | 69.4 | 69.4 | 49.4 | 0.002 | 0.025 | 24m |

**These numbers are essentially uninformative in absolute terms.** OCRBench V2 is a VQA benchmark — every sample has a per-sample question ("What is X in the image?") that a proper eval must feed as the prompt. `scripts/eval.py` uses a single fixed prompt, so all three models dump full page OCR instead of answering. The NED explodes because the "prediction" is orders of magnitude longer than the short expected answers.

Fixing this requires either (a) enhancing `scripts/eval.py` to route the JSONL's `prompt` field per-sample, or (b) vendoring OCRBench V2's official task-aware scorer (github.com/Yuliang-Liu/MultimodalOCR).

The relative ordering (iter-4 shortest outputs, base longest) is the only signal we can pull today. That signal confirms what OmniDocBench already said: our fine-tunes are more taciturn, less capable of English content generation.

## What this means for iter-7

The dominant finding is unambiguous:

> **Our Devanagari-focused fine-tuning has collapsed the base model's English/multilingual OCR capability.**

iter-2 shipped 30k mixed English/multilingual samples. iter-3 and iter-4 pivoted hard: 28k Devanagari word crops (iter-3) + 793 pseudo-labeled Devanagari pages (iter-4), with only ~500 English word replay samples for anti-forgetting. That anti-forgetting budget was **insufficient**: F1 on English pages collapsed from base's 0.291 to iter-4's 0.106, a 63% relative regression.

The `VISION.md` benchmark targets are English/multilingual (OCRBench V2 evaluates VQA in English + CJK; OmniDocBench is English-heavy academic content). We have optimized for Devanagari word-level accuracy at the cost of the very benchmarks that define the mission.

## Iter-7 = 7A (multilingual/English data expansion)

Per the plan file's iter-7 decision tree, this is Branch A: expand real training data toward VISION.md's stated 300k target. The prep scripts for DocVQA, TextOCR, HierText, XFUND, and IDL-WDS are all already committed under `scripts/prepare/`. Iter-7A's work is a new corpus mix + fresh LoRA r=32 on olmOCR-7B, 2 epochs, ~15h on g5.4xlarge, ~$25. Ship gate is OCRBench V2 F1 ≥ base olmOCR-7B's 0.025 while preserving Devanagari word CER ≤ iter-4's 28.1%.

The three other branches (7B structural, 7C Devanagari human labels, 7D GRPO) are all documented in the plan file but are secondary to Branch A given today's numbers.

## What this measurement iteration deliberately did NOT do

- **No new adapter training.** iter-6 is measurement, not modeling.
- **No 32B retry.** Still blocked on AWS quota (p4d/p4de need 96 vCPU P; approved is 64).
- **No competitor comparison** (Interfaze/Gemini/Claude/GPT API runs). Deferred — the strategic answer (Branch 7A) is decidable from our own numbers alone.
- **No OCRBench V2 per-sample-prompt enhancement.** Documented as a follow-up. Iter-7A can incorporate it.

## Artifacts

- `results/iter6/*.json` (6 files) — full per-sample results, synced to `s3://enclave-scribe-checkpoints/results/iter6/`
- `data/benchmark/omnidocbench_test.jsonl` (1,645 samples) and `data/benchmark/ocrbench_v2.jsonl` (10,000 samples) — synced to `s3://enclave-scribe-checkpoints/data/benchmark/`
- `reports/iter3/env_note.md`, `reports/iter4/env_note.md` — env-hygiene records from PR #58
- `configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml` — retired 32B design for future quota-unblocked rerun

## Non-negotiables carried into iter-7

- `reports/iter7/pip_freeze.txt` published with the adapter (mandatory per iter-5 postmortem).
- Staging `i-073a0fe419ceb9f49` never touched. Confirmed clean throughout iter-6.
- Every iter-N ships a decision paper (this one) regardless of adapter outcome.
- Honest reporting: our own fine-tuning made the model worse on the benchmarks the vision cares about. Published anyway.
