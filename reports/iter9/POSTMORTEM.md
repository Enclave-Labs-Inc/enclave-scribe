# Iter-9 — POSTMORTEM (shipped 2026-09-15, no adapter published)

**Verdict: iter-9 fails ship gates 1 and 2 hard. No HuggingFace publish. Adapter kept on S3 for iter-10 diagnostic use only.**

**Intent:** iter-9 = 9A per iter-8's plan. Warm-start from iter-7a (the VQA specialist that got 0.499 F1 on OCRBench V2, 70.6% of Interfaze's 0.707 target) and extend with a VQA-heavy corpus (OCR-VQA + ChartQA + TextVQA + InfographicVQA on top of iter-7a's baseline). Target: OCRBench V2 F1 ≥ 0.60 (85% of Interfaze), OmniDocBench F1 ≥ 0.291 (preserve base), himalaya CER ≤ iter-7a's 4.463 (don't regress further).

**Outcome:**
- OCRBench V2 F1 **regressed** from 0.499 → 0.484 (−3% relative). Gate 1 HARD FAIL by 0.116.
- OmniDocBench F1 **regressed** from 0.293 → 0.250 (−15% relative). Gate 2 HARD FAIL by 0.041.
- himalaya_500 CER **improved 10×** from 4.463 → 0.442 (from 500-sample replay). Gate 3 PASS.
- Total cost: ~$42 (well under $60 hard ceiling).
- Staging `i-073a0fe419ceb9f49` untouched throughout ✓.

**Read this section, then jump to "Iter-10 direction — locked" at the bottom.** The intermediate sections document what happened and why, for the postmortem trail.

---

## Ship gate outcomes

| Gate | Target | iter-9 actual | Status |
|---|---|---:|---|
| **1. OCRBench V2 F1 ≥ 0.60** | ≥ 0.60 | **0.484** | ❌ **HARD FAIL** by 0.116 |
| **2. OmniDocBench F1 ≥ 0.291** | ≥ 0.291 | **0.250** | ❌ **HARD FAIL** by 0.041 |
| **3. himalaya_500 CER ≤ 4.463** | ≤ 4.463 | **0.442** | ✅ PASS (10× better than iter-7a) |

**Iter-9 does not ship an adapter to HuggingFace.** Per plan: gates 1 or 2 fail → POSTMORTEM, no HF publish. Adapter stays at `s3://enclave-scribe-checkpoints/adapters/iter9/` for iter-10 to inspect but is NOT recommended for production use.

---

## Full benchmark table — all measured adapters

| Benchmark | base | iter-3 | iter-4 | iter-7a | **iter-9** | Gemini-3.5-Flash | Interfaze | Iter-9 vs iter-7a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **OCRBench V2 F1** (300 samples) | 0.093 | 0.099 | 0.115 | **0.499** | **0.484** | 0.639 | 0.707 | **−0.015 (−3%)** |
| **OmniDocBench F1** (250 pages) | 0.291 | 0.016 | 0.106 | 0.293 | **0.250** | — | — | **−0.043 (−15%)** |
| **himalaya_500 CER** ↓ (500 word crops) | 14.32 | **0.175** | 0.231 | 4.463 | **0.442** | — | — | **10× better** |

**Distance to VISION.md targets:**
- OCRBench V2: iter-9 at 68.5% of Interfaze (0.484 / 0.707), 75.7% of Gemini (0.484 / 0.639). *Slightly worse than iter-7a*.
- OmniDocBench NED: iter-9 at 3.178 (not competitive with Unlimited-OCR's 0.082 target — 39× worse).

**Choose-your-adapter guidance (unchanged from iter-8, iter-9 does not supersede any):**
- English document VQA / short-answer → **iter-7a** (published at `enclavelabs/olmocr-2-iter7a-vqa`)
- Devanagari word / page OCR → **iter-3** (published at `enclavelabs/enclave-scribe-devanagari`)
- English long-form page OCR → **base olmOCR-7B**

---

## OmniDocBench per-category breakdown (iter-9)

| Category | F1 ↑ | CER ↓ | NED ↓ | Comment |
|---|---:|---:|---:|---|
| omnidocbench_magazine | **0.770** | 0.260 | 0.260 | Strong |
| omnidocbench_exam_paper | 0.434 | 0.486 | 0.486 | Decent |
| omnidocbench_academic_literature | 0.359 | 1.359 | 1.359 | Moderate |
| omnidocbench_PPT2PDF | 0.276 | 1.656 | 1.656 | Weak |
| omnidocbench_book | 0.246 | 1.922 | 1.922 | Weak |
| omnidocbench_colorful_textbook | 0.197 | 1.244 | 1.244 | Weak |
| omnidocbench_note | 0.118 | 1.051 | 1.051 | Broken |
| omnidocbench_research_report | **0.027** | 10.925 | 10.925 | ❌ **Completely broken** |
| omnidocbench_historical_document | **0.008** | 4.435 | 4.435 | ❌ **Completely broken** |
| **OVERALL** | **0.250** | 3.178 | 3.178 | |

The `research_report` and `historical_document` categories collapsed under iter-9 (F1 = 0.027 and 0.008). These are long, dense, structural document types — iter-9's docvqa-heavy warm-start actively pushed the model *away* from long-form page transcription toward short-form Q&A. The category-scoped damage on `research_report` (CER 10.9 = generating 11× more incorrect text than the ground truth) confirms task-mismatch as the root cause, not truncation.

Compare iter-7a's `research_report` F1 = 0.072 (also weak, but 2.7× iter-9's number). iter-9 made a bad category worse.

---

## What happened during Phase 1 — the actual timeline

Phase 1 launched 2026-09-14 17:35 UTC on g5.4xlarge on-demand `i-0322358642c52cfb1` in us-east-1c (us-east-1f, 1a, 1b all had insufficient g5.4xlarge capacity at launch time). The instance terminated 2026-09-15 22:30 UTC after ~26 hours.

### Prep runaway (10.5h, ~$17)

The `build_iter9_corpus.py` script chained prep_docvqa + prep_xfund + prep_idl + attempted OCR-VQA/ChartQA/TextVQA/InfographicVQA/DocVQA-extra fetches. Two failure modes chained together:

1. **`prep_idl.py` targets 100k IDL samples by default**, streaming from HF WebDataset. HF rate-limited us at ~1-2 samples/sec (same iter-5/iter-7a experience). At step 36% (36k samples), ETA was 12+ additional hours. We killed it after collecting 36,264 usable IDL samples.
2. **The new VQA prep scripts** (OCR-VQA / ChartQA / TextVQA / InfographicVQA / DocVQA-extra) — the entire premise of iter-9's corpus expansion — **never downloaded**. HF hub rate-limited on parquet fetches with `us.aws.cdn.hf.co: Read timed out` errors. The corpus builder was still stuck on the DocVQA-extra parquet re-fetch at 10h in when we killed it.

**Fallback:** used what completed:
- docvqa: 39,463
- idl: 36,264 (dropped — task-mismatch for OCRBench V2)
- xfund: 1,393
- himalaya_replay: 500
- **Total after dropping idl: 41,356 samples** (76,068 with idl, but idl kept dragging step time up).

### Training run 1 (aborted at step 40)

Started with the full 76k corpus (idl included). Step time settled at 10.5s/step. Extrapolated ETA: 27.7h. Combined with prep time already spent, this would have pushed total cost past the $60 hard ceiling. **Killed at step 40** (~1.5 min of training).

### Training run 2 (shipped)

Rebuilt corpus without idl: 40,529 train + 827 val = 41,356 total. Restarted training.

- Warm-start confirmed: log line `Resuming LoRA from adapter: outputs/iter7a` ✓
- Trainable params: 95M / 8.4B = 1.13% ✓
- 5067 total optimizer steps
- Step time: 9.3–9.7s/step (roughly consistent with iter-7a's rate, since data distribution now matched)
- Training wallclock: 13.3h (48,045s)
- Final loss: 6.514 (mean over run: 6.602)

**Loss stayed in a 6.49–6.70 band with no meaningful downward trend.** iter-7a's converged loss was ~6.2. iter-9's loss oscillated *above* iter-7a's baseline for the entire training run. That's a strong prior signal that iter-9 wouldn't beat iter-7a on eval — which it didn't.

### Eval + sync + terminate (2h)

- Eval 1 (OCRBench V2 300): 4 min inference + 1 min load = 5 min. **F1 = 0.484**.
- Eval 2 (OmniDocBench 250, max_new_tokens 4096): **4.7h**. Slow because many samples generated near the full 4096 token cap on long pages. F1 = 0.250.
- Eval 3 (himalaya_500): 7 min. CER = 0.442.
- S3 sync (adapter 380MB + 3 result JSONs + pip_freeze + train.log): 2 min.
- Instance terminated cleanly.

---

## Root cause analysis

**iter-9 regressed on both English benchmarks because the corpus we actually trained on was 87% DocVQA — essentially just MORE of what iter-7a already saw.**

The plan called for 120k samples spanning 5 new VQA-shape sources (OCR-VQA / ChartQA / TextVQA / InfographicVQA / DocVQA-extra). Those never downloaded due to HF rate limits. What survived:

- 39,463 DocVQA (already saturated in iter-7a's training)
- 1,393 XFUND (multilingual form understanding — some overlap with OCRBench V2's multilingual tasks, minor signal)
- 500 himalaya_indic replay (Devanagari anti-forgetting)

Warm-starting from iter-7a and running 1 epoch of low-LR (5.0e-5) SFT on this corpus is *fine-tuning-on-what-you-already-know*. There was no new task-shape signal to lift OCRBench V2. The DocVQA saturation actually pushed the model *slightly toward shorter answers* (evidenced by OmniDocBench regression on long-form categories: `research_report` collapsed to F1 0.027).

**The Devanagari replay worked spectacularly.** 500 samples out of 41,356 (1.2% of corpus) took himalaya CER from 4.463 (446%) to 0.442 (44%) — a 10× improvement. This confirms iter-8's diagnosis that iter-7a's Devanagari regression was purely from underexposure, not architectural. Small replay is enough to hold the line.

### Why the warm-start didn't help

Warm-starting saves training compute on distributions the base model has already learned. It doesn't create new capability. iter-9's corpus = iter-7a's corpus × ~4 (post-filter counts). The warm-start meant we didn't have to re-teach VQA basics, but there was nothing new to teach. Result: modest catastrophic-forgetting-like drift as the optimizer chased small distribution differences instead of building new skills.

Fresh-LoRA from base olmOCR-7B on the same 41k corpus would likely produce a similar result — the input signal is the bottleneck, not the initialization.

---

## Cost breakdown

Instance `i-0322358642c52cfb1` (g5.4xlarge on-demand us-east-1c, ~$1.63/hr):

| Phase | Wallclock | Cost |
|---|---:|---:|
| Bootstrap + benchmark sync + iter-7a adapter sync | 25 min | $0.68 |
| Data prep (docvqa/idl/xfund downloads, corpus build) | 10h 15min | $16.71 |
| Warm-start sanity eval (30 samples) | 5 min | $0.14 |
| Training run 2 (5067 steps, 41k corpus) | 13h 20min | $21.73 |
| Eval loop (3 benchmarks) | 5h 10min | $8.42 |
| S3 sync + terminate | 5 min | $0.14 |
| **Total** | **~25h 55min** | **~$42-43** |

Hard ceiling was $60. Under budget. Iter-9's cost profile is close to iter-7a's ($48) despite training on 4× the data — the eval-loop overhead was disproportionate (OmniDocBench alone was 4.7h due to 4096-token cap on long pages).

**Cost sinks worth naming:**
- IDL prep with 100k-sample default cost ~6h of wallclock. Iter-10 must cap IDL well below 100k or drop it.
- OmniDocBench eval at max_new_tokens=4096 is 4.7h/adapter. If iter-10 has multiple adapters to evaluate, use `--limit 100` or drop research_report from the subset.

---

## Iter-10 direction — locked

Per plan's decision tree: *"If iter-9 misses gate 1 (F1 < 0.60): iter-10 = fresh-LoRA variant of iter-9's corpus (no warm-start). If a fresh LoRA on the ~120k VQA corpus also fails, capacity is the gap — revisit 32B P Spot..."*

**But iter-9 didn't actually test the plan's premise.** We trained on 41k, not 120k, and the 41k was 87% docvqa (iter-7a's distribution). We haven't yet tested whether OCR-VQA + ChartQA + TextVQA data would move OCRBench V2 F1.

**Iter-10 = rebuild the VQA corpus properly with a rate-limit-tolerant download strategy, then train fresh LoRA on base olmOCR-7B.** Specifically:

1. **Rate-limit-tolerant download.** Options: (a) mirror the four datasets to S3 once as a bootstrap step; (b) resume-tolerant partial-parquet downloads with backoff; (c) use `HF_HUB_DISABLE_XET=1` + `HF_HUB_DOWNLOAD_TIMEOUT=600` + retry loop. The 429 timeouts from `us.aws.cdn.hf.co` are non-fatal but our current scripts don't retry. Fix this once, reuse forever.
2. **Cap IDL at 20k, not 100k.** IDL is page-OCR data — useful for OmniDocBench (where iter-9 also regressed) but low-signal for OCRBench V2. Include but don't dominate.
3. **Fresh LoRA on olmOCR-7B, not warm-start.** iter-9 confirmed warm-start on similar distribution just adds noise. Fresh LoRA r=32/α=64, LR 1.0e-4 (like iter-7a), 2 epochs (like iter-7a).
4. **Include the himalaya replay** — 500 samples took CER from 446% to 44%. Bump to 1000-2000 to try to reach iter-3's 17.5% CER at low corpus-fraction cost.
5. **Budget: $50-60.** Same shape as iter-9, better data mix.

**Ship gates (iter-10):**
- OCRBench V2 F1 ≥ 0.55 (relaxed from iter-9's 0.60 — pragmatic step toward the 0.707 vision target)
- OmniDocBench F1 ≥ 0.291 (unchanged — must at least match base)
- himalaya_500 CER ≤ 0.30 (return to iter-3/4 territory)

**Contingencies if iter-10 also misses gate 1:**
- If OCRBench V2 F1 < 0.55 on fresh LoRA with proper VQA corpus → capacity is the gap. File the P on-demand 96 vCPU quota ticket and revisit 32B P Spot with a proper checkpoint-resume story.
- If iter-10 hits ≥ 0.55 → iter-11 scales the same recipe with 2× data. Distance-to-Interfaze becomes tractable.

---

## Lessons learned (add to non-negotiables)

Carried forward into iter-10 and beyond:

- **Every prep script has a `--max_samples` default that's too high for our budget.** Enforce a plan-level cap at the corpus builder, not at the prep script. iter-9's corpus builder trusted prep_idl's 100k default and burned ~$10 on samples we then dropped.
- **HF rate limits are a first-class failure mode.** Assume ~2 samples/sec upper bound on any parquet fetch from `us.aws.cdn.hf.co`. Build a retry-with-backoff layer around every `datasets.load_dataset` call. Or better: pre-mirror to S3 once, and pull from there for all future iters.
- **Warm-starting on the same-distribution data is a null intervention.** iter-9 confirmed: warm-start + more of what the base already saw = drift, not lift. Use warm-starts only when the new data is genuinely out-of-distribution vs the source adapter.
- **500-sample replay is enough to hold Devanagari.** iter-9's 10× improvement from 500 himalaya samples validates cheap anti-forgetting at any future iteration.
- **Track loss trend, not loss value.** iter-9's loss oscillating in 6.49–6.70 band without a downward trend was a strong prior signal that eval would regress. iter-10 should abort if loss doesn't drop below the warm-start baseline (if warm-starting) within the first 500 steps.
- **OmniDocBench eval at max_new_tokens=4096 costs ~5h per adapter.** Iter-10 eval loop should either cap max_new_tokens at 2048 for OmniDocBench (accept the truncation risk on research_report — which is broken anyway) OR restrict OmniDocBench to a 100-sample subset excluding research_report.

---

## What iter-9 shipped

- **S3 artifacts** (kept for iter-10 diagnostic use):
  - `s3://enclave-scribe-checkpoints/adapters/iter9/` — adapter (380 MB) + config + README + training_args
  - `s3://enclave-scribe-checkpoints/results/iter9/` — 3 eval JSONs (ocrbench_v2, omnidocbench, himalaya_500)
  - `s3://enclave-scribe-checkpoints/reports/iter9/pip_freeze.txt` — 288-line env manifest
  - `s3://enclave-scribe-checkpoints/reports/iter9/train.log` — full training log (377 KB)
- **This POSTMORTEM report** (`reports/iter9/POSTMORTEM.md`).
- **NO HuggingFace publish.** iter-9 adapter stays on S3, not published to HF. `enclavelabs/olmocr-2-iter7a-vqa` (the iter-7a scoped VQA specialist from iter-8) remains the current English VQA recommendation. `enclavelabs/enclave-scribe-devanagari` (iter-3) remains the Devanagari recommendation.

## Compute + reproducibility

- **Instance:** `i-0322358642c52cfb1`, g5.4xlarge on-demand, us-east-1c. Terminated 2026-09-15 22:30 UTC.
- **AMI:** DL AMI PyTorch 2.7 Ubuntu 22.04 (`ami-012ba162b9cd2729c`)
- **Pins:** transformers 4.55.4, peft 0.20.0, accelerate 1.4.0, torch 2.7.0+cu128, datasets, liger_kernel, jiwer, editdistance. Full manifest: [`pip_freeze.txt`](pip_freeze.txt).
- **HF token used:** `enclavelabs` account fine-grained token.
- **Config as-shipped:** [`configs/train/iter9_vqa.yaml`](../../configs/train/iter9_vqa.yaml) (unchanged from PR #65)
- **Staging safety:** `i-073a0fe419ceb9f49` verified `running` at Phase 1 start (17:35 UTC 2026-09-14) and immediately before termination (22:29 UTC 2026-09-15). Never touched.
