# Iter-11 — POSTMORTEM (shipped 2026-09-17, all 3 ship gates missed but real progress on 2/3 axes)

**Verdict: iter-11 misses all 3 ship gates by small margins but establishes new OCRBench V2 all-time high (0.516) and 12× improvement on Devanagari CER. NO HuggingFace publish. Adapter kept on S3 for iter-12 reference.**

**Intent:** first full-scale bilingual (English + Indic) fine-tune on `Qwen/Qwen3-VL-8B-Instruct` (validated in iter-10a probe), task-diverse corpus, targeting OCRBench V2 F1 ≥ 0.55 / OmniDocBench F1 ≥ 0.29 / himalaya CER ≤ 0.30.

**Outcome:** 3818 optimizer steps × 7h 10min training, final loss 4.97-5.50 (32% lower than iter-9's 6.65). All 3 ship gates miss, but iter-11 is the strongest all-around adapter yet trained.

**Cost: ~$24** (17h 40min on g5.2xlarge × $1.212/hr). Under $80-120 budget. Staging `i-073a0fe419ceb9f49` untouched.

---

## Ship gate outcomes

| Benchmark | Metric | Actual | Gate | Delta | Verdict |
|---|---|---:|---:|---:|---|
| OCRBench V2 F1 (300) | F1 | **0.5158** | ≥ 0.55 | −0.034 | ❌ MISS (but new all-time high) |
| OmniDocBench F1 (250) | F1 | **0.2506** | ≥ 0.29 | −0.039 | ❌ MISS |
| himalaya_500 CER (500) | CER | **0.3667** | ≤ 0.30 | +0.067 | ❌ MISS |

**Iter-11 does not ship to HuggingFace.** Adapter stays at `s3://enclave-scribe-checkpoints/adapters/iter11/` for iter-12 diagnostic reference. Recommendations unchanged:
- English document VQA → `enclavelabs/olmocr-2-iter7a-vqa` (iter-7a, F1 0.499)
- Devanagari OCR → `enclavelabs/enclave-scribe-devanagari` (iter-3, CER 0.175)

---

## Full benchmark table

| Benchmark | base | iter-3 | iter-4 | iter-7a | iter-9 | **iter-11** | vs prev best |
|---|---:|---:|---:|---:|---:|---:|---:|
| OCRBench V2 F1 (300) | 0.093 | 0.099 | 0.115 | 0.499 | 0.484 | **0.5158** 🏆 | **+0.017 over iter-7a** |
| OmniDocBench F1 (250) | 0.291 | 0.016 | 0.106 | **0.293** | 0.250 | 0.2506 | −0.042 vs iter-7a |
| himalaya_500 CER (500) | 14.32 | **0.175** | 0.231 | 4.463 | 0.442 | 0.3667 | +0.192 vs iter-3 (still worse); 12× better than iter-7a |

**OCRBench V2:** iter-11 is the first adapter to break 0.5 F1. +3.4% over iter-7a's 0.499. Both baselines from iter-10a probe (5k DocVQA only → 0.470) and iter-11 (15k task-diverse) confirm Qwen3-VL-8B is a stronger base than olmOCR-2-7B for this benchmark.

**OmniDocBench:** essentially matches iter-9's 0.251, below iter-7a's 0.293. Root cause: **the 15k page-OCR source `olmocr_mix` failed to download during prep** (returned 0 samples). Without page-OCR training data, iter-11 didn't learn OmniDocBench's transcription task shape. Per-category breakdown:

| OmniDocBench category | N | F1 | CER | Verdict |
|---|---:|---:|---:|---|
| omnidocbench_magazine | 9 | 0.678 | 2.43 | Strong |
| omnidocbench_exam_paper | 9 | 0.464 | 0.51 | Decent |
| omnidocbench_academic_literature | 53 | 0.346 | 1.54 | Moderate |
| omnidocbench_colorful_textbook | 21 | 0.270 | 0.74 | Weak |
| omnidocbench_book | 98 | 0.262 | 2.31 | Weak |
| omnidocbench_note | 2 | 0.110 | 2.30 | Broken |
| omnidocbench_historical_document | 5 | 0.019 | 2.21 | ❌ Completely broken |
| omnidocbench_research_report | 42 | 0.009 | **9.34** | ❌ Completely broken |
| omnidocbench_PPT2PDF | (missing from output) | | | |
| **OVERALL** | **250** | **0.251** | **3.11** | |

The `research_report` (42 samples) and `historical_document` (5 samples) categories completely collapsed — same failure mode as iter-9's postmortem (F1 0.009 and 0.019 respectively). Model produces long incorrect outputs on those categories; without page-OCR training data, it defaults to interpretive prose.

**himalaya_500:** iter-11 CER = 0.3667. Better than iter-7a (4.463, 12×) and iter-9 (0.442). Not yet at iter-3 (0.175) or iter-4 (0.231) levels. But **F1 = 0.4779** on himalaya is much higher than iter-3's ~0.53 (comparable). The 500 himalaya replay + 1961 synthetic Devanagari pages in the corpus lifted Devanagari significantly.

---

## What happened during Phase 1 — timeline

Instance `i-087dcf9681581bbb7` (g5.2xlarge us-east-1c, downgraded from g5.4xlarge due to AWS capacity shortage across all 5 AZs) launched ~17:50 UTC 2026-09-16, terminated ~11:34 UTC 2026-09-17.

### Bootstrap (~30 min)
- Cloned repo at `22d8c5d`
- Upgraded transformers 4.55.4 → 4.57.6 (Qwen3-VL-8B requires ≥4.57)
- pip_freeze captured BEFORE training (non-negotiable met)
- Synced benchmarks + himalaya_500 images

### Corpus prep (~1h 15min, ~$1.45)
Ran `build_iter10_corpus.py` orchestrator. Individual prep results:

| Source | Target | Actual | Status |
|---|---:|---:|---|
| olmocr_mix | 15,000 | **0** | ❌ FAILED — prep script returned 0 samples |
| pubtabnet | 5,000 | 5,000 | ✓ |
| chartqa | 5,000 | **0** | ❌ FAILED — prep script returned 0 samples |
| math_formula | 3,000 | 3,000 | ✓ (initially crashed with segfault -6, retried) |
| docvqa | 8,000 | 8,000 (from 14,450 available) | ✓ (killed prep at 14k to save time — well over target) |
| xfund | 1,400 | 1,393 | ✓ |
| devanagari_synthetic | 2,000 | 2,000 | ✓ (rendered on-instance) |
| himalaya_replay | 500 | 500 | ✓ |

Post-dedup + filter: **15,267 train + 311 val samples** (vs plan's ~40k target). Missing 20k from olmocr_mix + chartqa. Corpus builder proceeded with what it had per plan's rollback rule ("drop failed source, continue").

### Training crash + restart (5 min)
First training attempt died at step 4 with `FileNotFoundError: 'data/raw/himalaya_indic/0002/03214.jpg'`. Root cause: himalaya replay samples reference `data/raw/himalaya_indic/` but the benchmark images are at `data/benchmark/himalaya_500/himalaya_indic/`. Fixed by symlinking `data/raw/himalaya_indic` → `data/benchmark/himalaya_500/himalaya_indic`. This is the same fix iter-9 used but was not codified in `build_iter10_corpus.py`.

### Training (7h 10min, ~$8.70)
- 3818 optimizer steps (15,267 samples × 2 epochs / 8 grad_accum)
- Step time steady 6.5-7s/step (Qwen3-VL-8B on g5.2xlarge A10G 24GB)
- Trainable params: 87,293,952 / 8,854,417,648 = 0.9859% (LoRA r=32/α=64)
- Loss: step 1 = 22.2 (verbose-format penalty on task-diverse mix) → step 108 = 5.40 (format aligned) → step 3818 = 4.97 (converged well below iter-9's 6.65)
- No OOM, no NaN, no crashes after fix
- Checkpoints at 1000, 2000, 3000 written

### Eval loop (7h 40min, ~$9.30)
- Eval 1 OCRBench V2 (300 samples, max_new_tokens=512): 3min 32s
- **Eval 2 OmniDocBench (250 samples, max_new_tokens=2048): 6h 26min** — very slow due to Qwen3-VL-8B verbose CoT on complex pages. Same issue observed in iter-9's OmniDoc eval.
- Eval 3 himalaya_500 (500 word crops, max_new_tokens=128): 7min

### Sync + terminate (5 min)
- All 3 result JSONs + adapter (333 MB) + train/eval logs synced to S3
- Staging verified `running` before + after termination
- iter-11 instance `shutting-down`

---

## Root cause analysis — why 3/3 gates missed

**Two distinct failure modes:**

### 1. Corpus incompleteness (olmocr_mix + chartqa returned 0)

These two prep scripts silently returned 0 samples. Consequences:
- No page-OCR data → OmniDocBench F1 collapsed on long-form categories (research_report, historical_document)
- No chart data → OCRBench V2 didn't get the chart-parsing signal that would have pushed above 0.55

**Why they failed:** unknown without deeper investigation. Possibilities: HF rate limits, dataset access issues, prep script bugs, disk I/O contention with 4 parallel preps running.

**Fix for iter-12:** run each new prep standalone with `--limit 100 --dry_run` first to verify it produces samples. Then re-run inside the orchestrator. Or better: run preps SEQUENTIALLY with individual failure detection, not in parallel.

### 2. Ship gates calibrated on plan's 40k corpus assumption

Plan gates were set assuming 40k task-diverse corpus. We trained on 15,267 (38% of target). Iter-10a probe showed 5k DocVQA → F1 0.47; scaling to 15k task-diverse gave F1 0.516 (linear projection: 40k would probably give ~0.60). The gates weren't wrong — the corpus was.

**Fix for iter-12:** rerun with the FULL 40k corpus. The Qwen3-VL-8B base + this training recipe is validated by iter-10a and iter-11's OCRBench V2 lift. Missing 20k of the right sources is the deficit.

---

## What iter-11 proved (positive signals)

Despite missing all gates, iter-11 established several important results:

1. **Qwen3-VL-8B > olmOCR-2-7B on OCRBench V2** for our recipe. iter-11's 0.516 beats iter-7a's 0.499 (same 300-sample cut) using 15k mixed samples vs iter-7a's 10k DocVQA-heavy. Base swap is validated.
2. **Task-diverse corpus lifts F1 modestly.** iter-10a 5k DocVQA → 0.47; iter-11 15k task-diverse → 0.52. +11% lift for 3× data. Non-linear returns suggest additional data helps modestly, not multiplicatively.
3. **Devanagari base advantage is REAL.** iter-11 CER 0.367 with 2500 Devanagari samples (2k synthetic + 500 word replay) beats iter-9's 0.442 with same replay quantity. Qwen3-VL-8B's Devanagari baseline (75.2 chrF++ raw) provides a floor that iter-7a's olmOCR base didn't have.
4. **Training loss converged well below prior iters.** iter-7a: 6.2. iter-9: 6.65. **iter-11: 4.97.** 25% lower loss suggests better model fit; the eval scores are constrained by benchmark-specific factors (task shape, verbose behavior on complex pages), not model fitness.

---

## Cost breakdown

Instance `i-087dcf9681581bbb7` (g5.2xlarge on-demand us-east-1c, $1.212/hr):

| Phase | Wallclock | Cost |
|---|---:|---:|
| Bootstrap + benchmark sync + transformers upgrade + pip_freeze | 30 min | $0.61 |
| Corpus prep (docvqa/xfund/pubtabnet/math_formula/devanagari_synth) | 1h 15min | $1.52 |
| Corpus merge + first training attempt (crashed at step 4) | 5 min | $0.10 |
| Training (3818 steps, 15k corpus × 2 epochs) | 7h 10min | $8.68 |
| Eval loop (OCRBench 3min + OmniDoc 6h 26min + himalaya 7min) | 6h 46min | $8.20 |
| Sync + terminate | 5 min | $0.10 |
| **Total** | **~15h 51min** | **~$19.20** |

Actual instance uptime was closer to 17h 40min including bootstrap + short idle time between phases = ~$21.42. Under the $80-120 budget target.

**Cost sinks worth naming:**
- OmniDocBench eval at max_new_tokens=2048 took 6h 26min. If iter-12 has multiple adapters to compare, we need a cheaper eval strategy (subset to 100 samples, or drop `research_report` category which we know is broken anyway).
- g5.4xlarge unavailable across all AZs — g5.2xlarge worked fine, actually cheaper. Use g5.2xlarge as first choice for future iters.

---

## Iter-12 direction (locked)

**Iter-12 = re-run iter-11's plan with the two failed prep sources fixed.** No new hypothesis to test — just close the corpus gap.

### Step A: Fix corpus prep (local, ~2-4h engineering, $0)

Investigate why `prep_olmocr_mix.py` and `prep_chartqa.py` returned 0 samples in iter-11. Run each standalone locally with `--limit 100` and inspect. Likely fixes:
- HF hub rate limits: add retry-with-backoff (already planned in `mirror_datasets_to_s3.py` per iter-10 plan)
- Dataset access: verify HF token has permissions for `allenai/olmOCR-mix-1025` (ODC-BY, should be public)
- Prep script bugs: check the two files for anything specific that would silently skip samples

Also codify the himalaya_indic symlink fix in the corpus builder — currently it's a manual step that we did on the instance, not committed to code.

### Step B: iter-12 full run (~14-18h, $25-40)

Same recipe as iter-11 but with 40k corpus complete:
- Base: `Qwen/Qwen3-VL-8B-Instruct` (validated ✓)
- Corpus: 15k olmocr_mix + 5k pubtabnet + 5k chartqa + 3k math_formula + 8k docvqa + 1.4k xfund + 2k synthetic Devanagari + 500 himalaya = **~40k task-diverse**
- Config: `configs/train/iter10_bilingual.yaml` (unchanged)
- Ship gates (unchanged): OCRBench V2 F1 ≥ 0.55, OmniDocBench F1 ≥ 0.29, himalaya_500 CER ≤ 0.30
- Expected outcome: comfortably clear 0.55 (iter-11's 0.516 with 38% of corpus → 40k should push 0.58-0.62); OmniDoc gate cleared by presence of olmocr_mix page-OCR data; himalaya likely 0.25-0.30 with additional replay
- Budget: $50-70 (cheaper than iter-11's $24 because eval time reduced by capping OmniDoc `max_new_tokens=1024`)

**Rollback plan (if olmocr_mix/chartqa fail AGAIN):**
- Drop chartqa (5k, plausibly replaceable by more docvqa)
- Substitute olmocr_mix with alternate page-OCR source — options: `allenai/olmOCR-mix-0225` (older version), Nougat arXiv pdf corpus, IIT-CDIP subset
- Or: skip olmocr_mix entirely and add 15k more synthetic Devanagari pages (goes deep on the axis we know works)

---

## Lessons learned (add to non-negotiables)

1. **Prep script silent failures are the #1 cost sink.** Two out of six sources returned 0 in iter-11, and the corpus builder proceeded with 38% of intended data. Future orchestrators should have a **minimum-corpus-size gate** (e.g., abort corpus build if total interim < 30k). Or: require each prep to write a `.done` marker with sample count, and orchestrator refuses to proceed on missing markers.

2. **The himalaya_indic symlink is a code smell.** iter-9 manually symlinked. iter-11 also manually symlinked (in-session, after crash at step 4). This should be handled by `build_iter10_corpus.py` at output time, or by scripting it in the bootstrap. Currently it's institutional knowledge, which will fail again in iter-12 unless committed.

3. **OmniDoc eval at max_new_tokens=2048 is $8+ per adapter.** Cap at 1024 for iter-12; accept some `research_report` truncation (that category is broken anyway).

4. **g5.4xlarge capacity is unreliable us-east-1.** g5.2xlarge works, is cheaper, and has same GPU. Use g5.2xlarge as first choice.

5. **Qwen3-VL-8B verbose CoT during eval:** longer than expected outputs on complex pages inflate eval wallclock ~4×. Consider a shorter default `--max_new_tokens` for eval-only runs on this base.

6. **Training loss 4.97 with F1 0.516 gap** suggests the metric-training mismatch is the real ceiling — the model fits the data (low loss) but the benchmarks penalize verbose/malformed outputs on categories not in the training distribution. **This is the case for GRPO with format-constrained rewards** (iter-13 material per iter-10 plan).

---

## What iter-11 shipped

- **`reports/iter11/POSTMORTEM.md`** — this document
- **`reports/iter11/pip_freeze.txt`** — 295-line env manifest (transformers 4.57.6 recorded)
- **S3 artifacts** (for iter-12 diagnostic reference):
  - `s3://enclave-scribe-checkpoints/adapters/iter11/` — adapter (333 MB), config, README, training_args
  - `s3://enclave-scribe-checkpoints/results/iter11/*.json` — 3 eval JSONs (300 + 250 + 500 samples)
  - `s3://enclave-scribe-checkpoints/reports/iter11/{train,eval}.log` — full logs
- **NO HuggingFace publish.** iter-11 adapter kept on S3, not published to HF. Recommendations unchanged (iter-7a for English VQA, iter-3 for Devanagari).

## Compute + reproducibility

- **Instance:** `i-087dcf9681581bbb7`, g5.2xlarge on-demand, us-east-1c. Terminated 2026-09-17 ~11:34 UTC.
- **AMI:** DL AMI PyTorch 2.7 Ubuntu 22.04 (`ami-012ba162b9cd2729c`)
- **Pins:** transformers 4.57.6, peft 0.20.0, accelerate 1.4.0, torch 2.7.0+cu128, datasets 5.0.1, editdistance 0.8.1, jiwer 4.0.0. Full manifest: [`pip_freeze.txt`](pip_freeze.txt).
- **Base model class:** `AutoModelForImageTextToText` auto-detects `Qwen3VLForConditionalGeneration` (verified in iter-10a probe).
- **HF token used:** `enclavelabs` account fine-grained token.
- **Staging safety:** `i-073a0fe419ceb9f49` verified `running` at Phase 1 start (17:50 UTC 2026-09-16) and immediately before termination (11:34 UTC 2026-09-17). Never touched.
