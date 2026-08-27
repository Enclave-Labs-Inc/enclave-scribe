"""Single page-level prompt for the agentic OCR pipeline.

Design decision (2026-08-27): after benchmarking OLMoCR-2 and studying how
LlamaParse and Unsiloed actually work, we dropped per-region prompt routing
entirely. LlamaParse/OLMoCR use ONE unified page-level prompt on a whole
page image — the model is trained to handle every region type contextually.

The intelligence lives in the WEIGHTS, not in the prompt library. When we
fine-tune OLMoCR-7B on Indic + gazette data, the model learns to route
tables → HTML, images → descriptions, seals → identifications from
training examples, not from switching prompts at inference time.

If you're tempted to add per-region prompts back: read PLAN.md § "Where I
was wrong in the plan" first.
"""

PAGE = (
    "Extract this document page as clean markdown. "
    "Preserve the original script (Devanagari, Latin, or any other) exactly as it appears — "
    "do not transliterate or translate. "
    "For tables, use HTML with <table>, <tr>, <td>, <th>, and rowspan/colspan where cells merge. "
    "For images, figures, charts, seals, or logos, describe them inline in italics prefixed with "
    "'Figure:', 'Chart:', 'Seal:', or 'Logo:' as appropriate. "
    "For handwritten annotations, transcribe them exactly. "
    "Preserve reading order — top-to-bottom, left-to-right per column. "
    "Do not summarize, interpret, or add commentary. "
    "Output only the markdown — no preamble or explanations."
)
