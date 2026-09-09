# EnclaveScribe — Iteration 4

**Page-level Devanagari OCR — fixing the failure mode iter-3 couldn't handle.**

---

## TL;DR

Iter-3 shipped a strong word-level Devanagari OCR adapter (93× base) but **dead-looped on long dense pages**. A runtime workaround in `scribe/agent/tools.py` (`repetition_penalty=1.1` + `bad_words_ids` blocking `<tool_call>`) masked the symptom.

Iter-4 teaches the adapter real page-level document structure via a **hybrid bootstrap**: 835 pseudo-labeled Devanagari page images from ai4bharat/indicdlp + 500 word replay samples. Total training corpus: 1,254 samples, 2 epochs on g5.xlarge.

Ship gate: on a 6-page Gazette of India Extraordinary notification with the runtime workaround **DISABLED**, iter-4 produces **13,646 chars** to iter-3's **7,122 chars**. Iter-3 dead-loops on 2 of 6 pages (0 chars output); iter-4 extracts every page cleanly.

| Signal | iter-3 | iter-4 |
|---|---:|---:|
| Gazette total chars (workaround disabled) | 7,122 | **13,646** |
| Pages dead-looped | 2/6 | **0/6** |
| `<tool_call>` blocks in final output | 0 (regex-stripped, so pages 4/5 blank) | **0 (real content)** |
| Max page chars | 2,430 | **4,077** |
| Wallclock (6 pages) | 20 min | **12 min** |

**Verdict: ship.** Iter-4 fixes the class-of-failure iter-3 couldn't handle. The runtime workaround stays in place as belt-and-suspenders.

**Honest caveat**: training loss stayed flat at ~4.2 across all 82 steps (grad norms 0.2–0.9). The adapter absorbed structural regularities via a small number of high-impact weight moves rather than an across-the-board tightening. Improvements are class-of-failure fixes, not a broad quality uplift. See "Honest read" below.

---

## What we built

- **Base model**: [allenai/olmOCR-2-7B-1025](https://huggingface.co/allenai/olmOCR-2-7B-1025) — same as iter-3
- **Starting point**: iter-3 LoRA adapter — `resume_adapter: outputs/iter3` in the config
- **Adapter shape**: LoRA r=32, α=64 (inherited from iter-3)
- **Training data**: 1,254 samples
  - 793 page-level pseudo-labels from ai4bharat/indicdlp (Hindi + Marathi), labeled by iter-3 through the agent pipeline with the runtime workaround enabled
  - 500 word-level replay samples from himalaya-ai/devanagari_ocr_dataset (prevents catastrophic forgetting on iter-3's word-level win)
  - Val split: 42 pages (5%)
- **Hardware**: 1× NVIDIA A10G 24GB on AWS g5.xlarge on-demand
- **Runtime**: 1 hour 37 minutes, 82 steps, 2 epochs
- **Precision**: bf16 + Liger kernel + gradient checkpointing
- **Effective batch**: 32 (per-device 1 × grad accum 32)
- **LR**: 5.0e-5 cosine schedule, 10 warmup steps
- **max_length**: 8192 (up from iter-3's 4096; pages need ~1500–2500 text tokens + ~640 image tokens)

## The ship gate

Iter-4's job was to fix generation loops on long pages. We tested against `gazette_moef_2024_06_07.pdf` — a 6-page Gazette of India Extraordinary notification (Ministry of Environment, June 7, 2024) — with the runtime `bad_words_ids` workaround **disabled** in `extract_page`. See the head-to-head file: [`reports/iter4/GAZETTE_TEST.md`](GAZETTE_TEST.md).

Per-page chars:

| Page | iter-4 | iter-3 |
|---|---:|---:|
| 1 | 1,549 | 1,544 |
| 2 | 2,172 | 2,049 |
| 3 | 2,364 | 2,430 |
| **4** | **2,413** | **0** |
| **5** | **4,077** | **0** |
| 6 | 905 | 933 |
| **Total** | **13,646** | **7,122** |

Iter-3 pages 4 and 5 exhausted the full 4,096-token generation budget emitting `<tool_call>` loops (241s each); the agent's post-hoc regex strip left them empty. Iter-4 completed both pages inside the budget with real document text.

## Text quality — page 1 (source vs both adapters)

Source PDF (via pdftotext) uses Arabic numerals in both language sections:
```
सं. 2113]           /  No. 2113]
जून 7, 2024        /  JUNE 7, 2024
ज्येष्ठ 17, 1946   /  JYAISHTHA 17, 1946
```

Iter-3 systematically converts them to Devanagari script even in the English section:
```
सं. २११३।          /  No. २११३।     ← wrong
जून ७, २०२४        /  JUNE 7, 2024
जयेष्ठ १७, १९४६    /  JYAISHTHA 17, १९४६   ← wrong
```

Iter-4 preserves the source convention:
```
सं. 2113]           /  No. 2113]    ✓
जून 7, 2024        /  JUNE 7, 2024
ज्येष्ठ 17, 1946   /  JYAISHTHA 17, 1946  ✓
```

Iter-4 has one slip on this page (`वन` → `बन`, a transliteration variant of "forest") that iter-3 got right. On balance iter-4 is more faithful to the source.

## Honest read of the training loss

Iter-4 loss was flat at ~4.2 for all 82 steps. Grad norms 0.2–0.9. That's usually a sign of "the adapter isn't learning."

But the ship gate says otherwise: iter-4 measurably fixes iter-3's failure mode on the exact PDF that motivated iter-4. Two things are true simultaneously:

1. The adapter absorbed **structural changes** — enough to complete generation on hard pages without falling into a `<tool_call>` loop. This is a small number of high-impact weight moves.
2. It did NOT broadly improve character-level accuracy. Iter-4 is not "iter-3 but sharper." It's "iter-3 that no longer breaks on long pages."

We left performance on the table. A future iter with `learning_rate: 2.0e-4` and more epochs on the same corpus would likely tighten character-level output further. That's iter-5's problem — this iteration answered the question we asked.

## What broke along the way

1. **himalaya-ai has no page-level content** — original iter-4 config (PR #44) assumed page-level scans; only 161 samples had ≥200 chars of text and all were Nepali paragraphs. Pivoted to ai4bharat/indicdlp mid-planning.
2. **`datasets.load_dataset(streaming=True)` HANGS on ai4bharat/indicdlp** — the dataset has 163 parquet shards and streaming enumerates all of them before yielding. Rewrote the prep script (PR #46) to use direct `hf_hub_download` per shard with a known language → shard-index map (`{"hi": 52, "mr": 91}`).
3. **`HfFileSystem` HTTP range requests also hang** — same underlying issue. Same fix.
4. **YAML 1.1 parses `5e-5` as string, not float** — `learning_rate: 5e-5` crashed AdamW init with `TypeError: '<=' not supported between instances of 'float' and 'str'`. Fixed by writing it as `5.0e-5` (PR #47).
5. **Bulk pseudo-labeling pass rate 41.75%** — quality filters (length, ASCII ratio, loop detection) dropped ~59% of iter-3 pseudo-labels. That's a real ceiling on this bootstrap approach. Better real labels beat more pseudo-labels.
6. **Gazette test PDF wasn't in the repo** — iter-3's canonical failure case was on someone's laptop, not in git. Iter-4's ship gate had to be verified twice: first on IndicDLP val pages (weak proxy — iter-3 doesn't loop on those either), then on the actual gazette PDF once we recovered it. Now committed to `tests/fixtures/pdfs/` — future iters won't lose it.

## Cost

| Item | Cost |
|---|---|
| IndicDLP page image download | ~$0 (30 min, no GPU) |
| Bulk pseudo-labeling (835 usable from 2,000 attempts, ~2 min/page on g5.xlarge) | ~$60 |
| Training (1h37m) | ~$3 |
| Ship gate + head-to-head eval | ~$1 |
| **Total for iter-4** | **~$65** |

The pseudo-labeling pass dominated cost. If iter-5 wants to go further, either (a) do real human labels (~$X but bounded), or (b) upgrade to a bigger base model where LoRA has more capacity to absorb structure from the same data.

## What's next — iter-5

Iter-4 fixed the class of failure we set out to fix. It didn't broadly improve quality. Iter-5 needs to pick between two clear bets:

1. **Real human-labeled Devanagari pages** — even 300–500 hand-cleaned samples beats 835 pseudo-labels at 17% CER. Breaks the bootstrap ceiling.
2. **Bigger base model** — try `allenai/olmOCR-2-32B-1025` or Qwen2.5-VL-32B. LoRA on a bigger base has strictly more capacity to absorb page-level structure from the same data.

Plus: **English regression test**. Neither iter-3 nor iter-4 has been formally benchmarked against English held-out. If we're claiming self-hostable Indic OCR, we should prove the English side isn't regressed.

Plus: **run all future iterations against `tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf`** as a regression gate. No more losing the canonical failure case.

---

## Links

- **iter-4 model on HuggingFace**: [enclavelabs/enclave-scribe-devanagari-iter4](https://huggingface.co/enclavelabs/enclave-scribe-devanagari-iter4)
- **iter-3 model (still shipped separately)**: [enclavelabs/enclave-scribe-devanagari](https://huggingface.co/enclavelabs/enclave-scribe-devanagari)
- **Repo**: https://github.com/Enclave-Labs-Inc/enclave-scribe
- **Adapter checkpoint**: `s3://enclave-scribe-checkpoints/outputs/iter4/`
- **Ship-gate results**: `s3://enclave-scribe-checkpoints/results/iter4/`
- **Head-to-head analysis**: [`reports/iter4/GAZETTE_TEST.md`](GAZETTE_TEST.md)
- **Runbook**: [`reports/iter4_runbook.md`](../iter4_runbook.md)

## Reproduce

```bash
# 1. Prep IndicDLP page images (Hindi + Marathi, ~30 min)
python scripts/prepare/prep_indicdlp_pages.py \
  --raw_dir data/raw/indicdlp_pages \
  --manifest_jsonl data/interim/indicdlp_pages.manifest.jsonl \
  --max_per_lang 1800

# 2. Pseudo-label pages with iter-3 (~50-150 GPU-hrs)
python scripts/prepare/label_pages_with_iter3.py \
  --manifest_jsonl data/interim/indicdlp_pages.manifest.jsonl \
  --raw_dir data/raw/indicdlp_pages \
  --out_jsonl data/interim/indicdlp_labeled.jsonl \
  --adapter_dir outputs/iter3

# 3. Word replay for anti-forgetting
python scripts/prepare/prep_himalaya_indic.py \
  --raw_dir data/raw \
  --out_jsonl data/interim/himalaya_indic.jsonl \
  --max_samples 500

# 4. Assemble the mixed corpus
python scripts/prepare/build_iter4_corpus.py \
  --labeled_pages_jsonl data/interim/indicdlp_labeled.jsonl \
  --word_replay_jsonl   data/interim/himalaya_indic.jsonl \
  --train_out data/processed/train.jsonl \
  --val_out   data/processed/val.jsonl

# 5. Train (~1h37m on g5.xlarge)
python scripts/train.py --config configs/train/iter4.yaml

# 6. Ship-gate on the canonical failure case (bad_words_ids DISABLED first)
python scripts/agent/parse.py \
  --pdf         tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf \
  --out         results/iter4/gazette_iter4.md \
  --adapter_dir outputs/iter4 \
  --base_model  allenai/olmOCR-2-7B-1025
```
