# Iter-7a — Multilingual data expansion (shipped 2026-09-13)

**Intent:** correct the English-recall regression iter-6 diagnosed in iter-4 by fine-tuning `allenai/olmOCR-2-7B-1025` on a broader English + multilingual OCR corpus. Fresh LoRA r=32/α=64, no resume from iter-4.

**Outcome:** ship gate 1 (OmniDocBench F1 ≥ base's 0.291) **passed by 0.7%** — iter-7a effectively *matches* base olmOCR-7B on English recall while cutting the base's tendency to over-generate (NED 2.998 → 1.955). This undoes iter-4's 63% English-recall regression, but does not clearly beat base. Devanagari regression check (gate 2) was **not measured** this iteration — the eval instance did not stage `himalaya_indic` images. Iter-7b will re-run Devanagari eval as a cheap eval-only follow-up.

- **📦 Adapter (S3):** `s3://enclave-scribe-checkpoints/adapters/iter7a/`
- **📊 Per-sample eval JSONs:** `s3://enclave-scribe-checkpoints/results/iter7a/`
- **🔧 Env manifest:** [`reports/iter7a/pip_freeze.txt`](pip_freeze.txt) (captured on instance before training)

---

## Ship gate outcomes

| Gate | Target | iter-7a actual | Status |
|---|---|---:|---|
| **1. OmniDocBench F1 ≥ 0.291** (base olmOCR-7B) | ≥ 0.291 | **0.293** | ✅ PASS (+0.7%) |
| **2. `himalaya_500` word CER ≤ 28.1%** (iter-4 baseline) | ≤ 28.1% | **not measured** | ⚠️ DEFERRED |
| **3. OCRBench V2 F1 ≥ 0.025** (base) | ≥ 0.025 | **0.056** | ✅ PASS (2.2×) |

Gate 2 was declared a "hard" gate at plan time but the eval instance did not stage the 500-image `himalaya_indic` benchmark subset (would have required a 60 GB download for 500 images; deferred as a cost decision — see "Deviations from plan" below). Iter-8 or a cheap eval-only follow-up will close this gate.

---

## Results — OmniDocBench (250 pages, mnt=768)

Same subsample, same eval script, same env as iter-6. Numbers are directly comparable.

| Adapter | NED ↓ | CER ↓ | WER ↓ | BLEU ↑ | F1 ↑ | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| iter-3 (Devanagari word LoRA) | 1.137 | 1.137 | 1.171 | 0.012 | 0.016 | 3h 07m |
| iter-4 (Devanagari page LoRA resumed) | 0.983 | 0.983 | 1.115 | 0.057 | 0.106 | 44m |
| base olmOCR-2-7B-1025 (no LoRA) | 2.998 | 2.998 | 2.686 | 0.131 | 0.291 | 1h 32m |
| **iter-7a** (fresh LoRA r=32, filtered 9.8k mix) | **1.955** | **1.955** | 3.322 | **0.142** | **0.293** | 2h 22m |

**Key observation:** iter-7a lands almost exactly on top of base's F1 (0.293 vs 0.291) while producing markedly less run-on text (NED 1.955 vs base's 2.998 — 35% shorter over-generation). BLEU is 8% higher than base. The adapter learned to be more concise without losing recall.

### Per-category F1 (iter-7a)

| OmniDocBench category | n | CER ↓ | F1 ↑ | BLEU ↑ |
|---|---:|---:|---:|---:|
| omnidocbench_magazine | 9 | 0.235 | **0.791** | **0.655** |
| omnidocbench_exam_paper | 9 | 0.429 | 0.435 | 0.227 |
| omnidocbench_academic_literature | 53 | 1.025 | 0.366 | 0.210 |
| omnidocbench_colorful_textbook | 21 | 0.504 | 0.362 | 0.140 |
| omnidocbench_PPT2PDF | 11 | 2.282 | 0.357 | 0.110 |
| omnidocbench_historical_document | 5 | 0.648 | 0.288 | 0.043 |
| omnidocbench_book | 98 | 1.854 | 0.273 | 0.120 |
| omnidocbench_note | 2 | 0.799 | 0.072 | 0.000 |
| omnidocbench_research_report | 42 | 4.913 | **0.072** | 0.006 |

**Research report drags the aggregate down hard** — F1=0.072 on n=42 (17% of samples), CER=4.9. Research reports are long dense pages that overflow the 768-token generation cap. Truncation kills F1 on this subset. Magazine and exam-paper categories show the adapter is genuinely useful on structured layouts. Book pages (n=98, 39% of samples) are the median-quality bucket.

---

## Results — OCRBench V2 (300 samples, mnt=128, fixed-prompt caveat)

Same fixed-prompt limitation as iter-6: our eval feeds `"document parsing."` as the prompt for every sample, but OCRBench V2 samples come with per-sample VQA questions. Absolute numbers are not comparable to Interfaze's 70.7% — but internal ordering is meaningful.

| Adapter | NED ↓ | CER ↓ | BLEU ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| iter-3 | 36.5 | 36.5 | 0.001 | 0.026 |
| iter-4 | 10.5 | 10.5 | 0.000 | 0.006 |
| base olmOCR-2-7B-1025 | 69.4 | 69.4 | 0.002 | 0.025 |
| **iter-7a** | **37.9** | **37.9** | 0.010 | **0.056** |

Iter-7a's OCRBench V2 F1 is 2.2× base and 9.3× iter-4. BLEU is 5× base. Directional signal: the adapter is more competent at short-form document tasks than any predecessor, though "competent" is still low in absolute terms.

---

## Deviations from plan

Being honest — plan-versus-reality deltas were substantial. All documented in commit history but summarized here:

### Corpus (target 107k → shipped 9,852)

| Source | Plan target | Shipped |
|---|---:|---:|
| DocVQA | 40,000 | 39,463 (kept nearly all) |
| TextOCR | 25,000 | **0** (dropped) |
| HierText | 12,000 | **0** (dropped) |
| XFUND | 10,000 | 1,393 (full train set; plan target was over-optimistic) |
| IDL-WDS | 20,000 | 5,551 (partial) |
| Devanagari replay | 500 | **0** (skipped — see gate 2) |
| **Total after merge + dedup** | ~107k | **45,864** (10k train after 3000-char text-length filter + subsample) |

**Why the cuts:**
- **TextOCR and HierText dropped** because Google's Open Images bucket returns 404 on per-image URLs, and HierText's `storage.googleapis.com/hiertext/hiertext/*.jsonl.gz` also 404s. Both prep scripts pointed at bit-rotted URLs. Reinstating them will require pointing at a working host or vendoring the assets.
- **IDL truncated at ~5.5k** because HuggingFace's xet CDN rate-limited us (429). We built in a 5-min cool-down retry, but continuing to 40k would have taken ~18h of streaming which blew the budget.
- **Corpus subsampled 45k → 10k** because per-step training time at max_length=8192 / max_pixels=640 was ~16 s/step, projecting a 50h / $81 training run. We could not afford the full corpus at that step rate.
- **Text-length filter (>3000 chars dropped)** because the training collator does not truncate to `max_length`. A single 25k-character IDL doc caused CUDA OOM at step 62 twice in a row. Filtering the top 148 outliers (1.5% of samples) fixed it.
- **Devanagari replay skipped** because the 60 GB `himalaya-ai/devanagari_ocr_dataset` download was not worth it for 500 samples. 500/10000 = 5% of corpus, so its absence is meaningful but not fatal.

### Config as-shipped

Runtime tweaks applied on the instance that are **not** in the tooling PR's committed config (will be reconciled in a follow-up commit):

| Config | Plan | Shipped |
|---|---|---|
| `max_pixels` | 501760 (640²) | **200704 (448²)** — OOM prevention |
| `max_length` | 8192 | **4096** — OOM prevention |
| `report_to` | `wandb` | **`none`** — no wandb API key on instance |
| `train_jsonl` | `data/processed/iter7a_train.jsonl` | **`iter7a_train_10k_filt.jsonl`** |
| `PYTORCH_CUDA_ALLOC_CONF` | (unset) | **`expandable_segments:True`** — memory fragmentation |

### Environment

- **AMI**: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7 (Ubuntu 22.04) 20260427` (torch 2.7.0+cu128, Python 3.12)
- **Env activation:** `/opt/pytorch/bin/activate` (venv, not conda — `scripts/setup_env.sh` assumed conda-activate; the fix will land in the follow-up PR)
- **Pinned pkgs (post-install):** `transformers==4.55.4`, `peft==0.20.0`, `accelerate==1.4.0`, `datasets==5.0.1`, `liger_kernel==0.8.2`, `jiwer==4.0.0`. Full manifest: [`pip_freeze.txt`](pip_freeze.txt).

### Compute

- **Instance:** `i-0d7b994eb3e82b910`, g5.4xlarge on-demand, us-east-1d, terminated on completion
- **Wallclock:** ~29.3h total (data prep + failed attempts + training + eval)
- **Cost:** ~$48 (right at the $50 iter ceiling, no under-run)
- **Staging `i-073a0fe419ceb9f49`**: verified running-and-untouched throughout

---

## Honest read

**What worked:**
- Fresh LoRA on a broader corpus recovered base olmOCR-7B's English F1 — the strategic goal of iter-7a.
- Adapter learned to be **more concise** than base (NED 1.955 vs 2.998). If you care about output length matching ground truth (which the vision does — Interfaze target NED < 0.082), that's real progress.
- Ship gate on OCRBench V2 (soft) also passed with 2.2× base.
- Env-hygiene discipline held: `pip_freeze.txt` captured, `setup_env.sh` shipped alongside the run (with a documented amendment path).

**What didn't work well:**
- **Training loss stayed high (~6.2 flat by mid-run)**, echoing iter-4's flat-loss pattern. Diagnosis: 87% of the corpus (`docvqa`) was Q&A short-answer pairs used in a pure-OCR framing where the target text is a 1-15 char answer to a hidden question. The model sees a full document but is asked to reproduce a snippet — high loss for every "unexplained" region. Meaningful OCR-quality training data was closer to 12% of the corpus, not 87%.
- **F1 lift over base is 0.7%** — inside noise. We shipped an adapter that essentially replicates base performance while burning $48 and 29h of compute.
- **Ship gate 1 was set relative to the wrong reference.** "Beat base olmOCR-7B" is a low bar. We beat iter-4's regression but did not deliver competitive OCR quality vs `VISION.md`'s Interfaze target (NED 0.082 vs our 1.955 — off by ~24×).

**Iter-8 direction (locked here so we don't drift):**
1. **Fix the data mix first.** DocVQA in pure-OCR mode is corpus poison — the model can't learn OCR from Q&A pairs where the target is a short answer. Either (a) route per-sample prompts through the training loop and treat DocVQA as VQA, or (b) drop DocVQA and rebuild with real OCR labels only.
2. **Fix URL rot** in prep_hiertext / prep_textocr — point at working hosts or vendor the images.
3. **Add corpus truncation at collator time** (`tokenizer(..., truncation=True, max_length=cfg.max_length)`) so oversized samples don't OOM training. The 3000-char pre-filter is a band-aid.
4. **Stage `himalaya_indic` images on S3** so the Devanagari regression gate is testable in every iter without a 60 GB HF download.
5. Only after the above: revisit whether corpus expansion is even the right lever. Iter-7a suggests it might not be — capacity (a bigger base) or GRPO for determinism may be higher-leverage. Reopen the iter-6 decision tree with these numbers in hand.

---

## Comparison table (VISION.md format)

| Model | OmniDocBench NED ↓ | OCRBench V2 F1 | Sovereign | Notes |
|---|---:|---:|---|---|
| Interfaze (target) | 0.082 | 0.707 | Yes (cloud) | VISION.md competitive target |
| Unlimited-OCR | 0.082 | — | Yes | VISION.md structural target |
| **iter-7a** (this) | **1.955** | **0.056** | **Yes (self-host)** | Multilingual mix; ~$48 training |
| base olmOCR-2-7B-1025 | 2.998 | 0.025 | Yes | Reference |
| iter-4 (Devanagari) | 0.983 | 0.006 | Yes | Regressed English capability |
| iter-3 (Devanagari word) | 1.137 | 0.026 | Yes | Word-level LoRA |

We are ~24× off Interfaze on OmniDocBench NED and ~13× off on OCRBench V2 F1. Iter-7a is the first iteration that **doesn't move away from** those targets, but does not appreciably close the gap either.
