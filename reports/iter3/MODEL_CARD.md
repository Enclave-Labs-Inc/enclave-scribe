---
license: mit
library_name: peft
tags:
  - ocr
  - devanagari
  - hindi
  - indic
  - lora
  - qwen2.5-vl
  - vision-language
language:
  - hi
  - mr
  - sa
  - ne
base_model: allenai/olmOCR-2-7B-1025
pipeline_tag: image-text-to-text
datasets:
  - himalaya-ai/devanagari_ocr_dataset
metrics:
  - cer
  - wer
  - f1
model-index:
  - name: enclave-scribe-devanagari
    results:
      - task:
          type: image-text-to-text
          name: Devanagari OCR
        dataset:
          type: himalaya-ai/devanagari_ocr_dataset
          name: himalaya-ai Devanagari (500-sample held-out)
        metrics:
          - type: cer
            value: 0.174
            name: Character Error Rate
          - type: wer
            value: 0.468
            name: Word Error Rate
          - type: f1
            value: 0.534
            name: F1
---

# EnclaveScribe — Devanagari OCR (iter-3)

A LoRA adapter for [`allenai/olmOCR-2-7B-1025`](https://huggingface.co/allenai/olmOCR-2-7B-1025) that adds Devanagari OCR capability. The base model can transcribe English documents well but cannot read Devanagari at all — this adapter fixes that.

Built by [Enclave Labs](https://github.com/Enclave-Labs-Inc). MIT-licensed. Part of the [EnclaveScribe](https://github.com/Enclave-Labs-Inc/enclave-scribe) project — a self-hostable, Indic-first document OCR system.

## What it does

Given an image containing Devanagari text (Hindi, Marathi, Sanskrit, Nepali, Pali), returns the Unicode transcription. Best suited for **word-level and short-line images**. For full-page PDFs, use the [EnclaveScribe agent pipeline](https://github.com/Enclave-Labs-Inc/enclave-scribe) which handles page rasterization and generation-config tuning.

## Results

Evaluated on a 500-sample held-out slice of himalaya-ai/devanagari_ocr_dataset (never seen during training):

| Metric | Base OLMoCR-2-7B | **This adapter** | Improvement |
|---|---:|---:|---:|
| CER ↓ | 16.26 (1626%) | **0.174 (17.4%)** | **~93×** |
| WER ↓ | 22.64 | **0.468** | 48× |
| F1 ↑ | 0.013 | **0.534** | 41× |
| Latency ↓ | 1.75 s/sample | **0.84 s/sample** | 2× faster |

Base is unable to read Devanagari — it hallucinates verbose English descriptions instead of transcribing, which is why it's also slower (more tokens generated).

## How to use

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from PIL import Image

BASE = "allenai/olmOCR-2-7B-1025"
ADAPTER = "enclavelabs/enclave-scribe-devanagari"

processor = AutoProcessor.from_pretrained(BASE)
model = AutoModelForImageTextToText.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

image = Image.open("hindi_word.png").convert("RGB")
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": "Transcribe the Devanagari text from this image:"},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.1,   # prevents generation loops on long/dense inputs
    )

print(processor.batch_decode(
    out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
)[0].strip())
```

**Important**: use `repetition_penalty=1.1` (or higher) in generation. Without it, the model can enter degenerate loops on long or ambiguous inputs. See "Limitations" below.

## Training details

- **Base model**: `allenai/olmOCR-2-7B-1025` (Qwen2.5-VL-7B fine-tuned by Allen AI for English document OCR)
- **Method**: LoRA (r=32, α=64, dropout=0)
- **Trainable parameters**: ~95M of 8.4B (~1.1%)
- **Data**: 28,824 word-level Devanagari samples from [himalaya-ai/devanagari_ocr_dataset](https://huggingface.co/datasets/himalaya-ai/devanagari_ocr_dataset), covering Hindi, Marathi, Sanskrit, Nepali, Pali. Mix of printed and IIT-Indic-HW handwritten crops.
- **Hardware**: 1× NVIDIA A10G 24GB (AWS g5.xlarge on-demand)
- **Runtime**: 8.7 hours, 1 epoch, 901 steps
- **Precision**: bf16 + Liger kernel + gradient checkpointing
- **Effective batch**: 32 (per-device 1 × grad accum 32)
- **Optimizer**: AdamW, cosine LR schedule, LR 1.5e-4, 27 warmup steps
- **Compute cost**: ~$12

## Limitations

- **Word-level training data**: this adapter was trained on single-word Devanagari crops. It's strong on word/line-level transcription but weaker on full-page documents. For pages, use it via the [EnclaveScribe agent](https://github.com/Enclave-Labs-Inc/enclave-scribe) which rasterizes pages and applies generation-config fixes.
- **Generation loops on dense inputs**: without `repetition_penalty ≥ 1.1` in generation config, the model can enter loops that emit `<tool_call>` tokens (a Qwen2.5-VL quirk that survives `skip_special_tokens=True`) until `max_new_tokens` is exhausted. Always pass `repetition_penalty=1.1`.
- **English regression**: not measured against a held-out English benchmark. Qualitative testing on a bilingual Hindi/English gazette PDF suggests English is preserved, but this is not a formal claim.
- **Handwriting**: training data includes IIT-Indic-HW handwritten crops but performance varies by writer style.
- **Not evaluated on**: Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, other Indic scripts. This adapter is Devanagari-family only.

## Iter-4 plans

Iter-4 will add page-level Devanagari data (Nayana, IndicVisionBench) to close the page-level gap, and formally benchmark English regression. Follow [github.com/Enclave-Labs-Inc/enclave-scribe](https://github.com/Enclave-Labs-Inc/enclave-scribe) for updates.

## Citation

```
@software{enclavescribe_devanagari_2026,
  title  = {EnclaveScribe: Self-hostable Indic OCR — Devanagari adapter},
  author = {Enclave Labs},
  year   = {2026},
  url    = {https://huggingface.co/enclavelabs/enclave-scribe-devanagari},
}
```

## License

MIT — same as the [base model](https://huggingface.co/allenai/olmOCR-2-7B-1025) and the [EnclaveScribe repo](https://github.com/Enclave-Labs-Inc/enclave-scribe).
