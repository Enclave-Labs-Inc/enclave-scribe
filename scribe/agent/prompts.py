"""Per-region VLM prompts for the agentic OCR pipeline.

Each prompt is tuned for one region type. The orchestrator picks the prompt
based on the region label produced by layout detection.

Design notes:
- Prompts explicitly instruct the model NOT to summarize or interpret text.
  Fidelity > cleverness — the model must reproduce what it sees.
- Table prompt uses HTML (not markdown pipes) because HTML supports rowspan
  and colspan, which markdown pipes don't. Complex gazette tables need this.
- Image/seal prompts request short, structured descriptions so downstream
  markdown assembly stays clean.
- Indic (Devanagari, Tamil, Bengali, etc.) is handled by the base Qwen2.5-VL
  model — no per-language prompt is used, we trust the model to preserve script.
"""

# Extract all readable text from the region in the original script and layout.
TEXT = (
    "Extract all the text from this image exactly as it appears. "
    "Preserve line breaks, punctuation, and the original script (Devanagari, "
    "Tamil, Bengali, Latin, or any other). Do not translate. Do not summarize. "
    "Do not interpret. Output only the text — no explanations, no headings."
)

# Extract a table region as HTML, preserving spans.
TABLE = (
    "Extract this table as valid HTML using <table>, <thead>, <tbody>, <tr>, "
    "<th>, and <td> tags. Use rowspan and colspan attributes where cells "
    "merge across rows or columns. Preserve every cell's text exactly as it "
    "appears (any script, any language). Do not add or remove rows/columns. "
    "Output only the HTML table — no <html>, no explanations."
)

# Describe a figure, chart, diagram, or photograph.
FIGURE = (
    "Describe this figure in one to three sentences. If it is a chart or graph, "
    "state the chart type (bar, line, pie, etc.), the axis labels, and the "
    "key data trends. If it is a diagram or flowchart, describe the structure "
    "and the labels. If it is a photograph, describe what is shown. "
    "If the figure contains readable text (labels, legends, captions), "
    "transcribe that text exactly. Output only the description — no headings."
)

# Identify an official seal, emblem, logo, or stamp.
SEAL = (
    "Identify this seal, emblem, logo, or stamp. Give the official name of the "
    "entity it represents (e.g., 'State Emblem of India', 'Reserve Bank of "
    "India logo', 'Ministry of Environment stamp'). If any text is visible on "
    "the seal, transcribe it exactly. If you cannot identify the entity, "
    "describe what the seal depicts in one sentence. Output only the "
    "identification and any transcribed text — no headings."
)

# Extract data from a chart specifically (called when region is confidently a chart).
CHART_DATA = (
    "Extract the data from this chart as a markdown table with headers. "
    "Include the chart type (bar/line/pie/scatter) on the first line as: "
    "Chart type: <type>. Then include axis labels or category labels as table "
    "headers. Then include each data series as rows. If exact values are not "
    "readable, mark them as (approx). Output only the markdown table and the "
    "chart-type line — no explanations."
)

# Prompt registry — orchestrator uses this to look up prompt by region label.
BY_REGION = {
    "text":    TEXT,
    "table":   TABLE,
    "figure":  FIGURE,
    "seal":    SEAL,
    "logo":    SEAL,
    "chart":   CHART_DATA,
    "image":   FIGURE,   # generic image falls back to figure prompt
    "picture": FIGURE,
}


def for_region(label: str) -> str:
    """Look up the right prompt for a region label. Falls back to TEXT prompt."""
    return BY_REGION.get(label.lower(), TEXT)
