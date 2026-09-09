---
license: mit
library_name: peft
tags:
  - ocr
  - devanagari
  - hindi
  - marathi
  - indic
  - lora
  - qwen2.5-vl
  - vision-language
  - page-level
language:
  - hi
  - mr
  - sa
  - ne
base_model: allenai/olmOCR-2-7B-1025
pipeline_tag: image-text-to-text
datasets:
  - ai4bharat/indicdlp
  - himalaya-ai/devanagari_ocr_dataset
metrics:
  - cer
model-index:
  - name: enclave-scribe-devanagari-iter4
    results:
      - task:
          type: image-text-to-text
          name: Devanagari page-level OCR (ship-gate)
        dataset:
          type: gazette-of-india
          name: Gazette of India Extraordinary (6-page, Hindi + English)
        metrics:
          - type: chars
            value: 13646
            name: Total chars extracted (workaround DISABLED)
          - type: pages_looped
            value: 0
            name: Pages that dead-looped
---

# EnclaveScribe — Devanagari OCR (iter-4)

A LoRA adapter for [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025) that adds page-level Devanagari OCR capability. Iter-4 is fine-tuned on top of [iter-3](https://huggingface.co/enclavelabs/enclave-scribe-devanagari) with real Indic document page images plus word-level replay, and fixes iter-3's known dead-loop failure mode on long dense Hindi/Marathi pages.

Built by [Enclave Labs](https://github.com/Enclave-Labs-Inc). MIT-licensed. Part of the [EnclaveScribe](https://github.com/Enclave-Labs-Inc/enclave-scribe) project — a self-hostable, Indic-first document OCR system.

## What it does

Given an image containing Devanagari text (Hindi, Marathi, Sanskrit, Nepali, Pali), returns the Unicode transcription. Unlike [iter-3](https://huggingface.co/enclavelabs/enclave-scribe-devanagari) which was trained on word crops only, iter-4 also handles **full document pages** without falling into generation loops.

For full-page PDFs, use it via the [EnclaveScribe agent pipeline](https://github.com/Enclave-Labs-Inc/enclave-scribe) which handles page rasterization and generation-config.

## Results — page-level ship-gate

Tested against a 6-page Gazette of India Extraordinary notification (dense Hindi + English) with the runtime `bad_words_ids` workaround **disabled**:

| Signal | iter-3 | **iter-4** |
|---|---:|---:|
| Total chars extracted | 7,122 | **13,646** |
| Pages dead-looped | 2/6 | **0/6** |
| Max page chars (of 4,096 ceiling) | 2,430 | **4,077** |
| Wallclock (6 pages, g5.xlarge) | 20 min | **12 min** |

Iter-3 pages 4 and 5 exhausted the full generation budget emitting `<tool_call>` blocks and produced empty output after the post-hoc regex strip. Iter-4 completes every page inside the budget with real document text.

Iter-4 is also more faithful to source script conventions — for example, it preserves Arabic numerals inside English sections (where iter-3 hallucinates Devanagari digits like `२११३` instead of `2113`).

See the full head-to-head in [`reports/iter4/GAZETTE_TEST.md`](https://github.com/Enclave-Labs-Inc/enclave-scribe/blob/main/reports/iter4/GAZETTE_TEST.md).

## How to use

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

BASE = "allenai/olmOCR-2-7B-1025"
ADAPTER = "enclavelabs/enclave-scribe-devanagari-iter4"

processor = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

image = Image.open("hindi_page.png").convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Extract this document page as clean markdown. "
                                   "Preserve the original script exactly."},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
        repetition_penalty=1.1,   # belt-and-suspenders; iter-4 no longer strictly needs it
    )

print(processor.batch_decode(
    out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0].strip())
```

**Note**: iter-4 no longer dead-loops on the pages that broke iter-3, but we recommend keeping `repetition_penalty=1.1` in production. It costs nothing and stays as insurance.

## Training details

- **Base model**: `allenai/olmOCR-2-7B-1025`
- **Starting point**: iter-3 LoRA adapter (`enclavelabs/enclave-scribe-devanagari`), loaded via `resume_adapter` so training continues rather than starting fresh
- **Method**: LoRA (r=32, α=64, inherited from iter-3)
- **Trainable parameters**: ~95M of 8.4B (~1.1%)
- **Training data**:
  - **Pages** — 793 pseudo-labeled Devanagari page images from [ai4bharat/indicdlp](https://huggingface.co/datasets/ai4bharat/indicdlp) (Hindi + Marathi), labeled by iter-3 through the EnclaveScribe agent pipeline; quality-filtered from 2,000 attempts (41.75% pass rate)
  - **Word replay** — 500 word-level samples from [himalaya-ai/devanagari_ocr_dataset](https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_dataset) mixed in at a 5:1 pages:words ratio to prevent catastrophic forgetting on iter-3's word-level competency
  - **Total**: 1,254 training samples + 42 held-out val
- **Hardware**: 1× NVIDIA A10G 24GB (AWS g5.xlarge on-demand)
- **Runtime**: 1h37m, 2 epochs, 82 steps
- **Precision**: bf16 + Liger kernel + gradient checkpointing
- **Effective batch**: 32 (per-device 1 × grad accum 32)
- **Optimizer**: AdamW, cosine LR schedule, LR 5.0e-5, 10 warmup steps
- **max_length**: 8192 (up from iter-3's 4096; pages need ~1500–2500 text tokens + ~640 image tokens)
- **Compute cost**: ~$3 training only; ~$65 including the pseudo-labeling pass

## Limitations

- **Pseudo-label ceiling**: training labels came from iter-3, which has 17.4% CER on word crops. Iter-4 inherited character-level errors from those labels. Real human labels would break this ceiling in a future iteration.
- **Training loss stayed flat** at ~4.2 across all 82 steps. Iter-4's improvements are class-of-failure fixes (no more dead-looping) rather than a broad character-level uplift. Iter-4 is "iter-3 that doesn't break on long pages", not "iter-3 but sharper".
- **English regression**: not measured against a held-out English benchmark. Qualitative testing on a bilingual Hindi/English gazette PDF suggests English is preserved and Arabic numerals are handled correctly, but this is not a formal claim.
- **Not evaluated on**: Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, other Indic scripts. This adapter is Devanagari-family only.
- **Ship-gate scope**: verified on a 6-page bilingual gazette PDF. Longer documents, tables, and structure preservation not exhaustively benchmarked.

## Iter-5 plans

- **Real human-labeled pages** (300–500 samples) to break the pseudo-label ceiling, OR
- **Bigger base model** (e.g. `allenai/olmOCR-2-32B-1025`) with the same corpus for more LoRA capacity
- **English regression benchmark** for both iter-3 and iter-4
- **Regression gate**: every future iteration runs against `tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf` before shipping

Follow [github.com/Enclave-Labs-Inc/enclave-scribe](https://github.com/Enclave-Labs-Inc/enclave-scribe) for updates.

## Citation

```
@software{enclavescribe_devanagari_iter4_2026,
  title  = {EnclaveScribe: Self-hostable Indic OCR — Devanagari adapter (iter-4)},
  author = {Enclave Labs},
  year   = {2026},
  url    = {https://huggingface.co/enclavelabs/enclave-scribe-devanagari-iter4},
}
```

## License

MIT — same as the [base model](https://huggingface.co/allenai/olmOCR-2-7B-1025) and the [EnclaveScribe repo](https://github.com/Enclave-Labs-Inc/enclave-scribe).
