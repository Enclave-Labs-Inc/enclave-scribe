# EnclaveScribe — Vision, Goals & Strategy

**By Enclave Labs**

---

## What is EnclaveScribe?

EnclaveScribe is a sovereign, multimodal OCR and document intelligence system built and fine-tuned by Enclave Labs. It converts any document — scanned images, PDFs, PowerPoint decks, Word files, handwritten notes, multilingual text — into structured, machine-readable output using a single fine-tuned vision-language model.

The word *Enclave* is intentional: your documents never leave your infrastructure. No third-party API. No per-page billing. No data leaving your servers.

---

## The Problem

Today's document intelligence landscape forces a choice between accuracy and sovereignty:

- **Frontier model APIs** (GPT-5, Claude Sonnet, Gemini Flash, Grok) have strong general vision but are not optimised for OCR — scanned documents, low-quality images, dense tables, mixed languages. They also bill per token and require sending sensitive documents to external servers.
- **Specialised cloud OCR** (Interfaze.ai, Unlimited-OCR, AWS Textract, Google Document AI) deliver better OCR accuracy but are still cloud-only, per-page billing, and vendor-locked.
- **Open-source alternatives** (Tesseract, PaddleOCR, docling) are sovereign but fall behind on accuracy, especially for complex layouts and multilingual content.
- **Two-step pipelines** (e.g., docling + Ollama) work but are slow, brittle, and hard to maintain.

For enterprises handling legal documents, medical records, financial filings, or any regulated content — sending data to any external API is often a compliance non-starter. And paying per-page at scale is economically unsustainable.

---

## Vision

> **A single fine-tuned model that outperforms every frontier model and every specialised OCR API on document accuracy — running entirely on your own infrastructure, at zero per-query cost.**

EnclaveScribe is built on the belief that a purpose-trained, domain-specific model will always outperform a general-purpose frontier model on a specialised task. OCR is that task. We intend to prove it with published, reproducible benchmarks.

---

## The Benchmark Target

We benchmark on two industry-standard benchmarks:

### OCRBench V2 (primary — higher is better)

This is the benchmark Interfaze.ai used to show they beat every major frontier model. It is our primary competitive target.

| Model | OCRBench V2 | Sovereign |
|---|---|---|
| **EnclaveScribe** | **TBD** | **Yes** |
| Interfaze.ai | 70.7% | No |
| Gemini-3.5-Flash | 63.9% | No |
| Claude-Sonnet-5 | 59.2% | No |
| Gemini-3-Flash | 55.8% | No |
| Grok-4.3 | 54.7% | No |
| GPT-5.4-Mini | 52.7% | No |

**Target: > 70.7%** — beat Interfaze and every frontier model, as a sovereign on-premise system.

### OmniDocBench (secondary — lower NED is better)

| Model | NED | Sovereign |
|---|---|---|
| **EnclaveScribe** | **TBD** | **Yes** |
| Unlimited-OCR | 0.082 | No |
| GOT-OCR 2.0 | 0.143 | Yes |
| Qwen2.5-VL-7B (base) | 0.131 | Yes |
| DocOwl 1.5 | 0.198 | Yes |

**Target: < 0.082** — beat the current best.

---

## What We Want to Achieve

### 1. Beat frontier models on OCR accuracy

GPT-5, Claude, Gemini, and Grok are general-purpose models. A purpose-fine-tuned model trained exclusively on real document OCR data should outperform them on this specific task — and we will demonstrate this with published numbers.

### 2. Beat Interfaze.ai on OCRBench V2

Interfaze is the current #1 on OCRBench V2 at 70.7%. They use a proprietary hybrid CNN+transformer architecture. We use a fine-tuned open VLM. If we beat 70.7%, we beat the best specialised commercial OCR product on its own benchmark — a sovereign model that anyone can self-host.

### 3. Replace 2-step pipelines with a single model

The current Enclave pipeline uses docling for parsing and Ollama for understanding. EnclaveScribe collapses this into one model, delivering:
- **3-5x lower latency** per document
- **Single point of maintenance** — one model, one config, one serving stack
- **Better end-to-end accuracy** — no error propagation between stages

### 4. Full multilingual and multimodal coverage

- **Document types**: scanned images, digital PDFs, PPTs, DOCX, handwritten notes, tables, forms, mixed-layout documents
- **Languages**: English, Chinese, Japanese, Korean, French, German, Spanish, and more

### 5. Production-ready sovereign serving

- **SGLang** (~200 tok/sec on A100 80GB) — max throughput, n-gram repetition suppression
- **vLLM** (~150 tok/sec on A100 80GB) — OpenAI-compatible drop-in
- Zero external dependencies at inference time

---

## Strategy

### Iteration 1 — Establish the baseline

**Model:** Qwen2.5-VL-7B-Instruct  
**Method:** Full bf16 LoRA (r=128, alpha=256) — no quantization  
**Data:** ~300K real multilingual document samples (CORD, FUNSD, XFUND, HierText, TextOCR, IDL-WDS)  
**Hardware:** AWS EC2 g5.12xlarge (4x A10G 24GB), ~10 hours, ~$57  
**Benchmarks:** OCRBench V2 + OmniDocBench vs all competitors above

Even if we don't beat Interfaze in iteration 1, publishing honest numbers with a clear roadmap builds credibility. A transparent loss is more valuable than a selective win.

**Why no synthetic data:** Every training sample is a real document with real noise — varied fonts, low-quality scans, mixed languages, complex layouts. Synthetic data trains models to recognise synthetic patterns, not production challenges.

**Why no QLoRA:** Quantization during training degrades output quality in ways that compound across long documents. Unsloth gives 2x faster training and 60% less VRAM at full bf16 quality.

### Iteration 2 — Deterministic fine-tuning

After iteration 1 benchmark numbers:

1. **GRPO on top of the SFT checkpoint** — reinforcement learning that trains the model for consistent, structured outputs. This is what makes the difference between a model that *usually* gets it right and one that *always* gets it right. It directly targets Interfaze.ai's "deterministic" marketing claim.

2. **Structured output training** — bounding box coordinates + confidence scores alongside extracted text. This closes the feature gap with commercial providers.

3. **32B if needed** — if iteration 1 doesn't beat 70.7% on OCRBench V2, iteration 2 moves to Qwen2.5-VL-32B. Cost: ~$280 on the same hardware. Still within $500 total budget.

### Benchmarking as a PR strategy

Published, reproducible benchmarks are the most credible marketing in AI. The plan:

1. Run iteration 1 — train, eval, publish numbers
2. Run the same OCRBench V2 test set through Gemini, Claude, GPT, Grok, and Interfaze APIs
3. Publish a single comparison table: EnclaveScribe vs every competitor, same test set, same metrics
4. If we beat Interfaze — lead with accuracy + sovereignty
5. If we lose to Interfaze — lead with sovereignty, cost, and the iteration 2 roadmap
6. Every iteration is a new publication event

---

## Competitive Positioning

| | EnclaveScribe | Interfaze.ai | GPT/Claude/Gemini | Unlimited-OCR |
|---|---|---|---|---|
| Sovereign / on-premise | ✅ | ❌ | ❌ | ❌ |
| Per-query cost | $0 | ~$0.002/page | ~$0.01/page | Unknown |
| OCR-specialised | ✅ | ✅ | ❌ | ✅ |
| Open source | ✅ | ❌ | ❌ | ❌ |
| Custom deployment | ✅ | ❌ | ❌ | ❌ |
| OCRBench V2 | TBD | 70.7% | 52-64% | N/A |

---

## Success Metrics

- **Accuracy (OCRBench V2):** > 63.9% (beat Gemini Flash) in iteration 1; > 70.7% (beat Interfaze) by iteration 2
- **Accuracy (OmniDocBench NED):** < 0.131 (beat base Qwen2.5-VL-7B) in iteration 1; < 0.082 by iteration 2
- **Latency:** < 5 seconds per page at 300 DPI on A100 80GB
- **Throughput:** > 150 tok/sec on a single A100 with SGLang
- **Sovereignty:** Zero external API calls during inference

---

## Training Data

All data is real — no synthetic generation.

| Dataset | Samples | Coverage |
|---|---|---|
| CORD | ~11K | Receipts, structured forms |
| FUNSD | ~199 | Noisy scanned forms |
| XFUND | ~1,400 | 7 languages (ZH, JA, PT, ES, FR, IT, DE) |
| HierText | ~12K | Scene text, hierarchical structure |
| TextOCR | ~25K | Natural images with text |
| IDL-WDS | ~250K | Historical documents, diverse layouts |
| **Total** | **~300K** | Multilingual, multi-document-type |

---

*EnclaveScribe is built by Enclave Labs. Questions or collaboration: open an issue on GitHub.*
