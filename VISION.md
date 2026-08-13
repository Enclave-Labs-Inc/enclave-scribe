# EnclaveScribe — Vision, Goals & Strategy

**By Enclave Labs**

---

## What is EnclaveScribe?

EnclaveScribe is a sovereign, multimodal OCR and document intelligence system built and fine-tuned by Enclave Labs. It converts any document — scanned images, PDFs, PowerPoint decks, Word files, handwritten notes, multilingual text — into structured, machine-readable output using a single fine-tuned vision-language model.

The word *Enclave* is intentional: your documents never leave your infrastructure. No third-party API. No per-page billing. No data leaving your servers.

---

## The Problem

Today's document intelligence landscape forces a choice between accuracy and sovereignty:

- **Cloud OCR APIs** (AWS Textract, Google Document AI, Azure Form Recognizer, Interfaze.ai) deliver good accuracy but require sending sensitive documents to third-party servers, charge per page, and lock you into vendor pricing.
- **Open-source alternatives** (Tesseract, PaddleOCR, docling) are sovereign but fall behind on accuracy, especially for complex layouts, low-quality scans, and multilingual content.
- **Two-step pipelines** (e.g., docling + Ollama) work but are slow, brittle, and hard to maintain. Two models, two failure points, 3-5x the latency.

For enterprises handling legal documents, medical records, financial filings, government data, or any regulated content — sending that data to a cloud OCR provider is often a compliance non-starter.

---

## Vision

> **A single fine-tuned model that delivers state-of-the-art OCR accuracy on any document type, in any language, running entirely on your own infrastructure.**

EnclaveScribe is built on the belief that sovereignty and accuracy are not a tradeoff. With the right fine-tuning data, the right training methodology, and the right serving stack, an on-premise model can match or beat commercial cloud APIs — and we intend to prove it with published benchmarks.

---

## What We Want to Achieve

### 1. Beat the state of the art on OmniDocBench

OmniDocBench is the standard benchmark for document OCR, covering diverse document types: scientific papers, financial tables, handwritten notes, slides, web screenshots, and more.

Current leaderboard (NED — lower is better):

| Model | NED | Sovereign |
|---|---|---|
| Unlimited-OCR | 0.082 | No |
| GOT-OCR 2.0 | 0.143 | Yes |
| Qwen2.5-VL-7B (base) | 0.131 | Yes |
| DocOwl 1.5 | 0.198 | Yes |
| TextMonkey | 0.215 | Yes |

**Target: EnclaveScribe NED < 0.082** — beating the current best model, and doing it as a sovereign, on-premise system.

### 2. Compete with commercial providers on accuracy

We benchmark directly against Interfaze.ai and Unlimited-OCR using the same test set and the same metrics. We publish results honestly, including where we lose. Transparency is the foundation of credibility.

### 3. Replace 2-step pipelines with a single model

The current Enclave pipeline uses docling for document parsing and Ollama for LLM-based understanding. EnclaveScribe collapses this into one model, delivering:
- **3-5x lower latency** per document
- **Single point of maintenance** — one model, one config, one serving stack
- **Better end-to-end accuracy** — no error propagation between pipeline stages

### 4. Full multilingual and multimodal coverage

EnclaveScribe handles:
- **Document types**: scanned images, digital PDFs, PPTs, DOCX, handwritten notes, tables, forms, mixed-layout documents
- **Languages**: English, Chinese, Japanese, Korean, French, German, Spanish, and more — trained on XFUND (7 languages), HierText, and IDL-WDS (250K multilingual samples)

### 5. Production-ready serving

Two battle-tested serving backends out of the box:
- **SGLang** (~200 tok/sec on A100 80GB) — maximum throughput, custom n-gram logit processor for OCR repetition suppression
- **vLLM** (~150 tok/sec on A100 80GB) — OpenAI-compatible API, drop-in for existing tooling

---

## Strategy

### Iteration 1 — Establish the baseline (current)

**Model:** Qwen2.5-VL-7B-Instruct  
**Method:** Full bf16 LoRA (r=128, alpha=256) — no quantization compromise  
**Training:** Unsloth SFT on ~300K real document samples (CORD, FUNSD, XFUND, HierText, TextOCR, IDL-WDS)  
**Hardware:** AWS EC2 g5.12xlarge (4x A10G 24GB), ~10 hours, ~$57  
**Goal:** Publish honest benchmark numbers vs Unlimited-OCR and Interfaze.ai

Even if we don't beat Unlimited-OCR in iteration 1, publishing the numbers with a clear "what's next" roadmap is valuable. It establishes EnclaveScribe as a serious, transparent project — not vapourware.

**Why no synthetic data:** Synthetic data trains models to recognize synthetic patterns, not real-world document noise. Every training sample in EnclaveScribe comes from real documents with real OCR challenges: varied fonts, low-quality scans, mixed languages, complex layouts. This is what actually improves production accuracy.

**Why no QLoRA:** Quantization during training degrades output quality in ways that compound across long documents. This is a competitive benchmark project — accuracy is the entire point. Full bf16 LoRA with Unsloth gives 2x faster training and 60% less VRAM versus standard HuggingFace, with zero quality compromise.

### Iteration 2 — Deterministic fine-tuning (next)

After seeing iteration 1 benchmark numbers, we layer in:

1. **GRPO (Group Relative Policy Optimization)** on top of the SFT checkpoint — trains the model to produce consistent, structured outputs rather than just mimicking training text. This directly targets the "deterministic" capability that Interfaze.ai markets as their core differentiator.

2. **Structured output training** — bounding box coordinates + confidence scores alongside extracted text. This closes the feature gap with commercial providers that offer verifiable, localized extraction.

3. **Scale to 32B if needed** — if iteration 1 NED doesn't clear 0.082, iteration 2 moves to Qwen2.5-VL-32B. Training cost: ~$280 on the same hardware. Still within the $500 total budget when combined with iteration 1.

### Benchmarking as a PR strategy

Published, reproducible benchmark numbers are the most credible form of marketing in the AI space. The plan:

1. Run iteration 1 training and eval
2. Publish benchmark results on the GitHub repo and a public report
3. If the numbers are competitive — lead with accuracy
4. If we lose to Interfaze.ai on accuracy — lead with sovereignty, latency, and zero per-query cost
5. Use iteration 2 improvements as a follow-up story ("we shipped, here's what we changed, here's the improvement")

A loss published honestly is more credible than a win published selectively. The goal is a reputation for rigour, not spin.

---

## Competitive Positioning

| | EnclaveScribe | Interfaze.ai | Unlimited-OCR | Tesseract |
|---|---|---|---|---|
| Sovereign / on-premise | ✅ | ❌ | ❌ | ✅ |
| Per-query cost | $0 | $1.50-3.50/M tokens | Unknown | $0 |
| Multilingual (100+ langs) | ✅ | ✅ | Limited | Partial |
| All document types | ✅ | ✅ | ✅ | Limited |
| Custom logit processor | ✅ | ❌ | ❌ | N/A |
| Open source | ✅ | ❌ | ❌ | ✅ |
| OmniDocBench NED | TBD | TBD | 0.082 | >0.3 |

---

## Success Metrics

- **Accuracy:** OmniDocBench NED < 0.131 (beat base Qwen2.5-VL-7B) in iteration 1; NED < 0.082 (beat Unlimited-OCR) by iteration 2
- **Latency:** < 5 seconds per page at 300 DPI on A100 80GB
- **Throughput:** > 150 tok/sec sustained on a single A100 with SGLang
- **Coverage:** All document types and languages in the training set handled without format-specific code paths
- **Sovereignty:** Zero external API calls during inference — model runs entirely on customer hardware

---

## Training Data

All training data is real — no synthetic generation.

| Dataset | Samples | Coverage |
|---|---|---|
| CORD | ~11K | Receipts, structured forms |
| FUNSD | ~199 | Noisy scanned forms |
| XFUND | ~1,400 | 7 languages (ZH, JA, PT, ES, FR, IT, DE) |
| HierText | ~12K | Scene text, hierarchical structure |
| TextOCR | ~25K | Natural images with text |
| IDL-WDS | ~250K | Historical document library, diverse layouts |
| **Total** | **~300K** | Multilingual, multi-document-type |

---

*EnclaveScribe is built by Enclave Labs. Questions or collaboration: open an issue on GitHub.*
