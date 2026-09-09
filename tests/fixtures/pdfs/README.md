# Canonical failure-case PDFs

Test fixtures used as regression gates for every iteration of the OCR adapter.
These files are committed on purpose — losing them between iterations forced
iter-4 to be verified on a weaker proxy (see `reports/iter4/README.md`), so
they now live in-tree.

Every future iteration MUST run `scripts/agent/parse.py` against
`gazette_moef_2024_06_07.pdf` (with `bad_words_ids` disabled in
`scribe/agent/tools.py::extract_page`) and produce output that satisfies:

1. `grep -c "<tool_call>" <out.md>` = 0
2. No page hits `max_new_tokens` ceiling
3. `wc -c <out.md>` ≥ 12,000 (documented iter-4 baseline: 13,646 UTF-8 chars)
4. Text is coherent Devanagari + English, not repetition
5. Char count ≥ iter-4's baseline on the same document

## Fixtures

### `gazette_moef_2024_06_07.pdf` — PRIMARY ship-gate case

- **Source:** Gazette of India, Extraordinary, Part II Section 3 Sub-section (ii)
- **Issuer:** Ministry of Environment, Forest and Climate Change
- **Publication date:** 7 June 2024 (17 Jyaishtha 1946 Shaka)
- **Doc code:** CG-DL-E-07062024-254607
- **Pages:** 6
- **Content:** Dense Hindi + English mirrored notification on environmental clearance rules. Long paragraphs of legal-register Devanagari with intermixed English legal terms and section numbers.
- **Why this document:** Iter-3 dead-loops on pages 4 and 5 of this exact file (0 chars output, `<tool_call>` blocks stripped by post-hoc regex, `max_new_tokens=4096` exhausted). This makes it the canonical failure case for page-level generation on Devanagari.

### `gazette_moef_2006_09_14.pdf` — SECONDARY reference

- **Source:** Gazette of India, older environmental notification (SO 1533E, 14 September 2006)
- **Content:** Longer document, referenced by the 2024 primary fixture.
- **Purpose:** Held in reserve for future iterations that want a harder page-level test. Not currently gated on.

## Where to find them if these files ever get deleted

- The 2024 gazette (SO 2215): freely available on egazette.gov.in with document code `CG-DL-E-07062024-254607`
- The 2006 gazette (SO 1533E): egazette.gov.in, indexed under Ministry of Environment and Forests, 14 September 2006

## Storage

Kept as plain files in git. 2 MB total is small enough that git-LFS would add
more friction than value. Anyone cloning the repo gets the failure case for
free.
