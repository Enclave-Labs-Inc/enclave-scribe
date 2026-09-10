# Iter-6 pilot — human-labeled Devanagari pages

**Status:** planning. Do NOT execute until iter-5 lands (or is confirmed to fail).
**Drafted:** 2026-09-11

---

## Why this exists (independent of iter-5's outcome)

Iter-4's dry-run measured word-level Devanagari CER at **28.1% on 500 samples**, a **6.6-point regression** from iter-3's 21.5% on the same subset ([reports/iter5/DRYRUN.md](../iter5/DRYRUN.md)). The 500 word-replay samples mixed 5:1 with page pseudo-labels were not sufficient anti-forgetting protection.

Two scenarios point to iter-6:

- **If iter-5 (32B) succeeds** — great, but the pseudo-label ceiling still constrains any 7B adapter path. Iter-6 gives the community-hosted 7B a route forward without paying 32B inference cost.
- **If iter-5 (32B) fails** — the flat-loss pattern would then be confirmed as a *labels* problem, not a *capacity* problem. Iter-6 becomes the only way forward.

Either way, the work below is worth starting **now, in parallel** with waiting for iter-5's compute quota.

---

## Scope — 100–300 hand-cleaned Devanagari pages

Pilot size, not production. Goal is to prove:
1. **Quality delta**: can 200 human labels beat 800 pseudo-labels on the same held-out gate?
2. **Labeling workflow**: what does a page-level Devanagari OCR label look like, and how long does it take?
3. **Cost per label**: is a full pilot feasible? Rough estimate before pilot: $2-5 per page × 300 = $600-1500.

**Non-goal**: production coverage. That's iter-7+ if the pilot signal is real.

---

## Data source

Same seed pool as iter-4: `ai4bharat/indicdlp` Hindi + Marathi pages. Prep script already exists at [`scripts/prepare/prep_indicdlp_pages.py`](../../scripts/prepare/prep_indicdlp_pages.py).

**Selection strategy for the pilot**:
- Start from the 793 pages iter-4 ended up training on (they passed iter-4's quality filters — ASCII ratio, length, no loops)
- Sample 300 at random with `random.seed(42)` for reproducibility
- Hand-label those exact 300, so we have a **direct head-to-head** with iter-4's pseudo-labels on the same images

Alternative: pull 300 fresh pages iter-4 never saw. But then we can't compare "labels quality" cleanly — we'd be mixing "different labels" with "different images".

**Decision: pilot uses the same 300 images iter-4 trained on.**

---

## Labeling workflow options

### Option A — Vendor (Scale AI, Surge, Labelbox)

- Pros: turnaround days-not-weeks, quality guaranteed, no infra
- Cons: expensive ($5-15/page for careful multi-lingual OCR), IP surface, minimum-project fees usually kill hobby projects
- Timeline: 2-4 weeks (contracting + labeling)
- **Cost estimate: $1500-4500 for 300 pages**

### Option B — Community labelers (Hugging Face, Prolific, Upwork)

- Pros: cheaper ($1-3/page), direct communication
- Cons: quality variance is real; need spot-check protocol; need multiple labelers per page for consensus on unclear text; language competence gate is critical
- Timeline: 3-6 weeks (recruitment + labeling + QA)
- **Cost estimate: $600-1800 for 300 pages** including QA overhead

### Option C — Bilingual friends/community + you as reviewer

- Pros: cheapest, quality control you can trust
- Cons: bounded by personal network; can burn favor; no repeatability if the pilot succeeds
- Timeline: 4-8 weeks
- **Cost estimate: $0-500 (thank-you gifts, not paid labor)**

### Option D — LLM-assisted with human review (hybrid)

- Pros: iter-3 or GPT-4o generates a candidate, human only corrects — 3-5× throughput
- Cons: anchoring bias — reviewer may miss errors the model made confidently
- Timeline: 1-2 weeks
- **Cost estimate: $300-900 for 300 pages** (reviewer time + LLM API)

**Recommendation for pilot: Option D (LLM-assisted with human review).** Prompt iter-3 via the existing agent pipeline, dump the output alongside the page as pre-populated text, hand to human reviewer with clear "these are draft transcripts; correct every error you see" framing. Cheapest path to a real quality signal.

---

## Labeling tool

Two viable options for Option D workflow:

- **Prodigy** (~$490 one-time license) — best OCR interface, side-by-side image + editable text, keyboard shortcuts, active-learning ready
- **Custom Streamlit app** (~1-2 days to build) — free, works exactly how we want, no license question

**Recommendation: build the Streamlit app.** 300 pages doesn't justify Prodigy's ROI for a pilot, and we already have the infra pieces (pdf2image, PIL, transformers) in the venv.

Rough shape (~200 lines):
```python
# scripts/label/review.py
# Streamlit UI, loads pre-populated iter-3 transcript, image side-by-side,
# writes {image, text, reviewer, reviewed_at, iter3_original} to
# data/interim/iter6_labels.jsonl
```

---

## Success gate for the pilot

The pilot succeeds if a 7B LoRA (r=32, otherwise identical to iter-3 config) trained on the 300 human labels beats iter-4's 28.1% word CER AND iter-3's 21.5% word CER on the fixed `data/benchmark/himalaya_500.jsonl` regression set.

**Concrete threshold: iter-6 pilot ships as `iter6-pilot` on HF if word CER ≤ 18%.**

That would be a 3.5-point improvement over iter-3 — clearly attributable to label quality, not other variables. Below that, we'd say "pilot inconclusive, more labels needed."

---

## Timeline (once labeling starts)

| Week | Milestone |
|---|---|
| 1 | Streamlit review tool built + iter-3 pre-labels generated for 300 pages |
| 2-3 | Human reviewers work through pages (3-5 pages/hour est.) |
| 3 | Quality-audit: spot-check 30 random pages, measure inter-annotator agreement on 20 shared pages |
| 4 | 7B LoRA training + eval on `himalaya_500.jsonl` |
| 4 | Decide: ship, keep labeling, or abandon |

Total: **~4 weeks from labeling kickoff**.

---

## Cost estimate (full pilot)

| Item | Cost |
|---|---|
| Streamlit tool build | $0 (time) |
| iter-3 pre-labeling pass (300 pages × ~2 min) | ~$5 GPU |
| Human reviewer time (300 pages × 15 min × $30/hr) | ~$2,250 |
| 7B training on g5.4xlarge (2-3 hrs) | ~$5 |
| Ship-gate eval on same instance | ~$5 |
| **Total pilot** | **~$2,265** |

If we use Option C (community/friend labelers), pilot drops to **~$300-500**. That's the pragmatic default unless someone's willing to fund the vendor path.

---

## Non-goals

- Multi-language pilot (Tamil, Telugu, Bengali) — iter-6 is Devanagari-only. Multi-script expansion is iter-7+.
- Structural/layout preservation labels — pilot only annotates text content. Tables and figures stay as "not annotated" for the pilot.
- Real-time labeling UI — batch-mode Streamlit is fine.

---

## Open questions (must answer before pilot kickoff)

1. **Labeler recruitment path** (A/B/C/D above)?
2. **Budget ceiling** — how much can we spend on the pilot before it becomes not worth the signal?
3. **Reviewer language competence gate** — how do we screen for actual Devanagari literacy (not just Hindi-speakers-but-not-readers)?
4. **What happens to iter-5's model on HF** if the pilot succeeds — does iter-6-pilot supersede iter-5, or coexist?

Get answers before writing `scripts/label/review.py`.
