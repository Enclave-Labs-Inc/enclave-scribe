# Iter-5 postmortem — no adapter shipped, iter-6 becomes the path

**Date:** 2026-09-11
**Status:** No iter-5 adapter shipped. Iteration ships as postmortem.
**Total spend:** ~$5 across two instances (g5.12xlarge spot ~$3, g5.4xlarge on-demand ~$2). Under the plan's $15 budget cap.
**Instances terminated:** `i-0ad434d6c8ef443f0` (g5.12xlarge), `i-0b73a9f8e654c5009` (g5.4xlarge). Staging `i-073a0fe419ceb9f49` untouched throughout.

---

## What was supposed to happen

Iter-4's Devanagari word CER regressed to 28.1% (vs iter-3's 21.5%) and training loss stayed flat at ~4.2. The iter-5 plan (2026-09-10) tested one hypothesis: was iter-4 capped by pseudo-label quality or by base-model capacity? Original design: fresh LoRA on a bigger olmOCR base. Mid-execution the plan pivoted to two 7B controlled runs (iter5a — Qwen2.5-VL-7B fresh, isolates OCR pretraining; iter5b — olmOCR-7B r=64, isolates LoRA capacity).

Neither shipped.

## What actually happened — three independent walls

### Wall 1: `allenai/olmOCR-2-32B-1025` does not exist

The 2026-09-10 plan referenced a repo that AllenAI has never published. Only `allenai/olmOCR-2-7B-1025` and quantized variants exist. Substituting `Qwen/Qwen2.5-VL-32B-Instruct` (the base olmOCR-7B was distilled from) works, but that model is not OCR-pretrained — so any null result would be ambiguous.

The retired config sits at `configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml` with a header explaining why.

### Wall 2: 32B + FSDP + LoRA does not fit 4× A10G 24GB

On `i-0ad434d6c8ef443f0` (g5.12xlarge spot, 4× A10G 24GB, 186 GB CPU RAM), six crashes in ~90 min:

- GPU OOM at model load — each rank loaded full 64 GB weights before FSDP could shard (each A10G is 22 GB usable).
- Rank-0-only CPU init with `fsdp_cpu_ram_efficient_loading` — CPU OOM at ~120 GB per rank × 4 ranks > 186 GB.
- LoRA + FSDP + activation checkpointing config conflicts (both `activation_checkpointing` in FSDP config and `gradient_checkpointing` in HF training args being on).

FSDP + CPU offload would have added ~3× wallclock and pushed past the $50 budget cap. When P-instance or g5.48xlarge quota lands (4 AWS support cases still open at time of writing), the 32B design is executable from the attic config with clean VRAM headroom.

### Wall 3: 7B fine-tuning environment drift, iter-4's shipped stack lost

On `i-0b73a9f8e654c5009` (g5.4xlarge on-demand, 1× A10G 24GB — the exact hardware iter-4 shipped on), 9+ failures within ~2 hours:

- **iter5a (Qwen2.5-VL-7B-Instruct)** OOM'd at 20.13 GB PyTorch alloc even at max_pixels=384×28×28, max_length=4096, no liger-kernel, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The Instruct-tuned variant has a heavier per-forward-step footprint than olmOCR-7B (which is the same base, distilled for OCR). Not runnable on 24 GB for LoRA fine-tuning.
- **iter5b (olmOCR-7B, same base as iter-4)** also OOM'd at ~20 GB under `transformers==4.55.4`, even at iter-4's exact r=32 / max_pixels=640 / max_length=8192. Iter-4 ran here without issue in early Sept 2026 — so transformers 4.55 has a memory regression vs whatever iter-4 shipped on.
- **Version archaeology hit a wall.** Every earlier transformers release we tried had a Qwen2.5-VL-specific bug:
  - `transformers==4.46.3` — `qwen2_5_vl` model_type not registered (added in 4.49).
  - `transformers==4.49.0` and `4.50.0` — `AttributeError: 'dict' object has no attribute 'to_dict'` in `GenerationConfig.from_model_config`.
  - `transformers==4.52.4` — `TypeError: argument of type 'NoneType' is not iterable` on `ALL_PARALLEL_STYLES` during `post_init`.
  - `transformers==4.55.4` — the memory regression documented above.
- **liger-kernel** was equally noisy: 0.8.2 needs `torch.distributed.tensor.DTensor` (torch ≥ 2.5, we're on 2.4.1); 0.4.2 imports `_CONFIG_FOR_DOC` from `transformers.models.gemma` (removed in 4.55.x).

Without iter-4's exact `pip freeze` recorded (the repo's `pyproject.toml` has only lower bounds), rediscovering the working combination is expensive dice-rolling. We paused after ~2 hours to avoid burning further into the budget on env archaeology.

---

## What we learned (still useful, even without an adapter)

1. **The plan's assumed target model didn't exist.** Iter-5 as scoped 2026-09-10 was structurally impossible; the pivot to Qwen2.5-VL-32B was the right substitution but ran into wall 2. This is a plan-review lesson: verify referenced HF repos actually exist before scheduling compute.
2. **The 32B path needs real headroom, not creativity.** FSDP-CPU-offload on 4× 24 GB is a fool's errand at 32B — even if you get it working, the wallclock cost eats any spot-price savings. When quota arrives, do it once, cleanly.
3. **Qwen2.5-VL-7B-Instruct is materially heavier than olmOCR-2-7B-1025.** Both are ~7B params; the Instruct variant's activation footprint pushes past 24 GB even at heavily-reduced input dims. Any future iter comparing them needs A40/A100/H100.
4. **We need to freeze `pip freeze` outputs.** iter-4 shipped, we can't run its config on the same hardware two months later. That's a repo-hygiene failure. Recording iter's env alongside its adapter is now iter-6 non-negotiable.

## Strategic implication — the labels vs capacity question is still open

Iter-5 was supposed to disambiguate iter-4's flat-loss ceiling. It didn't run. So:

- **Corpus (pseudo-label) ceiling hypothesis** — unfalsified; still the leading interpretation of iter-4's regression.
- **LoRA capacity ceiling hypothesis** — unfalsified; we didn't get to test r=64 for a full epoch, let alone 4.
- **OCR-pretraining bias hypothesis** — unfalsified; Qwen2.5-VL-7B-Instruct comparison didn't run.

In the absence of iter-5 evidence, the pragmatic move is to **treat pseudo-labels as the working hypothesis and pursue iter-6 human labels**. If iter-6's pilot beats iter-4's 28.1% CER on the same held-out gate, labels were the ceiling and the iter-5 32B run — if we ever do it — becomes an interesting secondary experiment rather than the critical path.

## Next steps

**Immediate (this PR):**
- Publish this postmortem.
- Update `README.md` roadmap: iter-5 postmortem shipped, iter-6 elevated to critical path.
- Retire `outputs/iter5a/SKIPPED.md` marker (on the instance, not committed) — describes the same wall as this postmortem.

**Iter-6 kickoff (blocked on labeler decision, not on code):**
- Pick labeler track from `reports/iter6/PILOT.md` (A vendor / B community / C friends / D LLM-assisted). Streamlit review tool is already committed at `scripts/label/review.py`.
- Set a labeler budget ceiling ($300-$2,250 depending on track).
- Confirm HF terms accepted for the 300 pilot pages already selected from iter-4's IndicDLP set.

**Deferred (blocked on AWS):**
- 4 support cases pending for P-instance / g5.48xlarge quotas. When any lands, the 32B design at `configs/train/attic/iter5_g5_32b_failed_2026-09-11.yaml` is executable in a single clean run. Reopen as iter-8 or later; not on iter-6's critical path.

## Non-negotiables carried into iter-6

- Every training run publishes its `pip freeze` next to the adapter. `outputs/<iter>/pip_freeze.txt` is a required artifact.
- Verify referenced HF repos exist before writing them into a plan.
- Keep the staging instance (`i-073a0fe419ceb9f49`) untouched. Reconfirmed today.
- Postmortems ship regardless of outcome. Honest reporting includes negative findings.
