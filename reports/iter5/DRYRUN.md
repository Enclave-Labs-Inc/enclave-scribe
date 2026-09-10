# Iter-5 dry-run — regression baselines + scaffolding findings

**Date:** 2026-09-10
**Instance:** g5.4xlarge on-demand (`i-0ba172f9d053a9937`, terminated after run)
**Spend:** ~$10-12
**Purpose:** validate `scripts/eval_regression.py`, establish iter-3/iter-4 baselines before iter-5 training

---

## TL;DR

Iter-4 **regressed word-level Devanagari CER by 6.6 percentage points** vs iter-3. This is a first-time head-to-head — iter-4 shipped without measuring word-level regression, and the number turned out worse than expected. The 500 word-replay samples mixed 5:1 with page-level pseudo-labels weren't enough anti-forgetting protection for iter-3's core competency.

| Metric | iter-3 | iter-4 | Δ |
|---|---:|---:|---:|
| Devanagari CER ↓ | **0.2151** | **0.2808** | +0.0657 ❌ regression |
| Devanagari WER ↓ | 0.4491 | 0.4544 | +0.0053 ~flat |
| Devanagari NED ↓ | 0.2151 | 0.2808 | +0.0657 ❌ regression |
| Devanagari F1 ↑ | 0.5609 | **0.5976** | +0.0367 ✅ better |
| Devanagari BLEU ↑ | 0.0352 | 0.0353 | +0.0001 flat |
| n_samples | 500 | 500 | – |
| elapsed (7B eval) | 430s | 428s | – |

Both adapters got the first two samples identical (`क्रिस्तिआन`, `कूलबेथ`); iter-4 loses on the harder ones.

## Two additions to iter-5's plan

### 1. Hard gate: iter-5 word CER must be ≤ 21.5%

The original iter-5 plan (in `~/.claude/plans/vectorized-dreaming-hearth.md`) gates on Devanagari word CER ≤ 20%. That threshold assumed iter-4's word CER was ≤ iter-3's. This dry-run reveals it's actually *worse*. Iter-5 must at minimum not regress further from iter-4's 28.1%, and ideally recover iter-3's 21.5% baseline. **If iter-5 word CER on `data/benchmark/himalaya_500.jsonl` is > 21.5%, do not ship.**

### 2. Eval-scaffolding gap: full-page English inference is not viable at 7B

The harness ran fine on Devanagari word crops (~1 sec/sample) but choked on English OmniDocBench pages at **200+ sec/sample** (~28 hours per 500-sample run). The model is asked to generate full markdown for a whole document page; token counts run into the thousands per response.

Before iter-5 uses this harness for English regression, `scribe.infer.local.infer_image` needs a **generation budget cap** (e.g., `max_new_tokens=512` for the eval path) and probably a **max_pixels cap** at the eval boundary. Alternatively, iter-5's English test set should be pre-cropped to smaller regions (~1 paragraph each), matching the crop scale of the Devanagari word test.

This is a real iter-5 blocker for English regression, filed as a scaffolding fix rather than a training question.

## What we set aside

- **English regression eval** on OmniDocBench/CORD/FUNSD/XFUND heldout — abandoned mid-run once per-sample latency became clear (see above)
- **DocVQA prep** — killed at 72% through its full training split (unnecessarily processed; DocVQA test split only needs a few thousand samples, not the whole 40k train set)

## How to reproduce

```bash
# On a GPU box with the venv set up:
export LD_LIBRARY_PATH=$(python -c 'import site,os;p=site.getsitepackages()[0];print(":".join([os.path.join(p,"nvidia/cudnn/lib"),os.path.join(p,"nvidia/cublas/lib")]))'):$LD_LIBRARY_PATH

# Rebuild the 500-sample Devanagari eval JSONL from himalaya-ai prep output
python scripts/prepare/prep_himalaya_indic.py \
    --raw_dir data/raw --out_jsonl data/interim/himalaya_indic.jsonl \
    --max_samples 500

python -c "
import json
with open('data/interim/himalaya_indic.jsonl') as fi, open('data/benchmark/himalaya_500.jsonl','w') as fo:
    for line in fi:
        r = json.loads(line)
        fo.write(json.dumps({'image': r['image'], 'text': r['text'], 'category': 'himalaya_indic'}, ensure_ascii=False) + '\n')
"

for adapter in iter3 iter4; do
    python scripts/eval.py \
      --gt_jsonl data/benchmark/himalaya_500.jsonl \
      --image_root data/raw \
      --base_model allenai/olmOCR-2-7B-1025 \
      --adapter_dir outputs/$adapter \
      --out_json reports/iter5/${adapter}_devanagari_dryrun.json
done
```

## What broke along the way (worth documenting for iter-5 setup)

1. **`flash-attn>=2.6.0` in `train` extras** blocks a fresh venv install because flash-attn's setup.py imports `torch` before it's installed. Workaround: install base + `eval` extras, then add `peft accelerate datasets` explicitly, skipping flash-attn. (Eval doesn't need flash-attn; `sdpa` attention is fine.)
2. **cuDNN 9.x runtime libraries** aren't on `LD_LIBRARY_PATH` by default with a `uv`-managed venv. torch's ctypes loader looks for `libcudnn_engines_runtime_compiled.so.9.25.x` on the system path but doesn't auto-expand the pip-installed wheel's lib dir. Fix: `export LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH` (also cublas). Symptom without the fix is an easily-missed warning during first inference plus 100% CER because `scripts/eval.py`'s try/except swallows the failure.
3. **`scripts/eval.py:52-55`** silently converts *any* per-sample exception to `pred=""`. On this dry-run, that masked both a FileNotFoundError (wrong image paths) AND the cuDNN failure. Recommend surfacing the first exception per run to stderr instead of blindly swallowing — trivial change, big debuggability win. Filed but not fixed in this branch.
4. **`prep_himalaya_indic.py --max_samples 500`** downloads the first 500 samples from the dataset's default ordering. This is NOT the same 500-sample subset that iter-3's original eval used. Baselines reported here (21.5% CER on iter-3) are on our subset, not iter-3's original subset (17.4% CER). Neither is "wrong" — they're different draws from the same distribution. For iter-5's regression gate, we standardize on THIS subset (`data/benchmark/himalaya_500.jsonl`, seeded via `--max_samples 500`).
5. **g5.xlarge OOM'd** during the first attempt at English data prep — the earlier prep pipeline built an in-memory index for 7.5M items on a 16GB RAM box. Fixed by moving to g5.4xlarge (64GB RAM) and skipping training-only datasets (TextOCR, HierText, IDL) that don't feed the eval heldout anyway.

## Artifacts

- `reports/iter5/iter3_devanagari_dryrun.json` — 500 per-sample preds from iter-3
- `reports/iter5/iter4_devanagari_dryrun.json` — 500 per-sample preds from iter-4
- S3: `s3://enclave-scribe-checkpoints/results/iter5/regression_v2/`
- Eval JSONL committed: none (build via reproduction steps above)
