"""Synthetic full-page Devanagari documents — replaces iter-9's 500-word-crop replay.

WHY THIS EXISTS
    iter-9's 500-sample himalaya_indic replay (word crops) achieved a 10× CER
    lift (446% → 44%), proving anti-forgetting is cheap. But per arXiv 2510.19546
    (Ali et al., 2025), replay works best when *distributionally matched* to what
    the source model has seen. Our base — Qwen3-VL-8B — expects full-page
    documents, not isolated word crops. This module generates full-page
    synthetic Devanagari documents so replay matches the training distribution.

APPROACH (minimal viable — Phase 0 goal is "renders legible Hindi text on a
full page", not photo-realism)
    1. Load Hindi text corpus (default: bundled sample; production: AI4Bharat).
    2. Compose 15-25 lines of Hindi text per page, wrap into paragraphs.
    3. Render onto a 1280x1600 white/beige/gray background using Pillow +
       Noto Sans Devanagari.
    4. Apply mild degradation (gaussian noise, tiny rotation, slight blur).
    5. Save PNG + write JSONL row {image, text, source}.

FONT SOURCE
    Noto Sans Devanagari (SIL Open Font License 1.1) — downloaded once to
    data/fonts/NotoSansDevanagari-Regular.ttf. If missing, script fetches it
    via https://github.com/notofonts/devanagari.

TEXT SOURCE (in order of preference)
    1. `data/interim/hindi_corpus.txt` if present (user-supplied).
    2. `ai4bharat/indic-instruct-data-v0.1` filtered to Devanagari (via datasets).
    3. Built-in fallback corpus of 200 common Hindi sentences.

OUTPUT
    data/raw/devanagari_synthetic/pages/NNNNNN.png
    data/raw/devanagari_synthetic/pages/NNNNNN.txt
    data/interim/devanagari_synthetic.jsonl (rows point to the images)

USAGE
    # First-time (also downloads font if missing)
    python scripts/prepare/synth_devanagari_pages.py --n 2000

    # Probe mode — render 3 pages to /tmp, print filenames, no JSONL write
    python scripts/prepare/synth_devanagari_pages.py --dry_run_probe --out /tmp/synth_probe/

DEPENDENCIES
    Pillow (bundled with our stack), datasets (optional, for AI4Bharat pull).
    NO dependency on SynthTIGER — we render directly with Pillow to avoid
    the SynthTIGER pip-install path (which has flaky OpenCV dependencies).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]

FONT_DIR = REPO_ROOT / "data" / "fonts"
FONT_PATH = FONT_DIR / "NotoSansDevanagari-Regular.ttf"
FONT_URL = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/"
    "main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Regular.ttf"
)

# Small built-in fallback so the script works with zero network + zero corpus
FALLBACK_HINDI_LINES = [
    "भारत एक विशाल देश है।",
    "हिंदी भारत की राजभाषा है।",
    "दिल्ली भारत की राजधानी है।",
    "गंगा नदी उत्तर भारत में बहती है।",
    "हिमालय पर्वत भारत के उत्तर में स्थित है।",
    "मैं अपने विद्यालय जा रहा हूँ।",
    "यह पुस्तक बहुत रोचक है।",
    "बच्चे पार्क में खेल रहे हैं।",
    "आज मौसम बहुत सुहावना है।",
    "किसान खेत में काम कर रहा है।",
    "सूर्य पूर्व दिशा में उदय होता है।",
    "चंद्रमा रात में चमकता है।",
    "गाय दूध देती है।",
    "फूल बहुत सुंदर है।",
    "पेड़ हमें छाया देते हैं।",
    "पक्षी आकाश में उड़ते हैं।",
    "मछली पानी में रहती है।",
    "शेर जंगल का राजा है।",
    "अध्यापक विद्यार्थियों को पढ़ाते हैं।",
    "स्वास्थ्य सबसे बड़ा धन है।",
    "मेहनत का फल मीठा होता है।",
    "समय बहुत मूल्यवान है।",
    "पानी जीवन के लिए आवश्यक है।",
    "पुस्तकालय ज्ञान का भंडार है।",
    "कंप्यूटर आधुनिक युग की देन है।",
]


def _ensure_font() -> Path:
    if FONT_PATH.exists():
        return FONT_PATH
    FONT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Noto Sans Devanagari → {FONT_PATH}", flush=True)
    r = subprocess.run(
        ["curl", "-fSL", "-o", str(FONT_PATH), FONT_URL],
        check=False,
    )
    if r.returncode != 0 or not FONT_PATH.exists():
        raise SystemExit(
            f"Failed to download font from {FONT_URL}. "
            f"Manually place a Devanagari TTF at {FONT_PATH} and re-run."
        )
    return FONT_PATH


def _load_hindi_corpus(min_lines: int = 500) -> list[str]:
    """Try file → HF dataset → built-in fallback."""
    local = REPO_ROOT / "data" / "interim" / "hindi_corpus.txt"
    if local.exists():
        lines = [l.strip() for l in open(local, encoding="utf-8") if l.strip()]
        if len(lines) >= min_lines:
            print(f"Loaded {len(lines)} Hindi lines from {local}")
            return lines

    # Try HF (best-effort; if it fails, fall through to built-in)
    try:
        from datasets import load_dataset
        print("Trying ai4bharat/indic-instruct-data-v0.1 (Hindi subset) ...",
              file=sys.stderr, flush=True)
        ds = load_dataset("ai4bharat/indic-instruct-data-v0.1",
                          "hi", split="train", streaming=True)
        lines: list[str] = []
        for i, sample in enumerate(ds):
            if i >= 5000:
                break
            for key in ("instruction", "input", "output", "text"):
                v = sample.get(key)
                if v and isinstance(v, str) and any(ord(c) >= 0x0900 and ord(c) <= 0x097F for c in v):
                    for line in str(v).split("\n"):
                        line = line.strip()
                        if 10 <= len(line) <= 200:
                            lines.append(line)
        if len(lines) >= min_lines:
            print(f"Loaded {len(lines)} Hindi lines from AI4Bharat")
            return lines
    except Exception as e:
        print(f"  ai4bharat pull failed ({e}); using built-in fallback",
              file=sys.stderr, flush=True)

    # Built-in fallback (looped to reach min_lines)
    print(f"Using built-in {len(FALLBACK_HINDI_LINES)}-line fallback (looped)",
          file=sys.stderr, flush=True)
    return (FALLBACK_HINDI_LINES * (min_lines // len(FALLBACK_HINDI_LINES) + 1))[:min_lines]


def _render_page(
    text_lines: list[str],
    font_path: Path,
    rng: random.Random,
    width: int = 1280,
    height: int = 1600,
) -> tuple["PIL.Image.Image", str]:
    """Render one synthetic Devanagari page. Returns (image, ground_truth_text)."""
    from PIL import Image, ImageDraw, ImageFont, ImageFilter

    # Random background tone (white / light beige / light gray)
    bg_color = rng.choice([(255, 255, 255), (250, 245, 235), (245, 245, 245), (252, 250, 240)])
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Random font size to add variety
    font_size = rng.randint(28, 40)
    font = ImageFont.truetype(str(font_path), font_size)

    # Compose 15-25 lines
    n_lines = rng.randint(15, 25)
    chosen = rng.sample(text_lines, min(n_lines, len(text_lines)))
    if len(chosen) < n_lines:
        # Pad by cycling
        while len(chosen) < n_lines:
            chosen.append(rng.choice(text_lines))

    # Wrap lines to fit width
    margin_x = rng.randint(60, 120)
    y = rng.randint(60, 120)
    line_spacing = font_size + rng.randint(6, 14)

    ground_truth: list[str] = []
    max_char_width = int((width - 2 * margin_x) / (font_size * 0.55))

    for raw in chosen:
        wrapped = textwrap.wrap(raw, width=max_char_width) or [raw]
        for wl in wrapped:
            if y + line_spacing >= height - 60:
                break
            # Very slight jitter for realism
            x_jitter = rng.randint(-2, 2)
            text_color = rng.choice([(0, 0, 0), (10, 10, 10), (25, 25, 40)])
            draw.text((margin_x + x_jitter, y), wl, fill=text_color, font=font)
            ground_truth.append(wl)
            y += line_spacing
        if y + line_spacing >= height - 60:
            break

    # Mild degradation — apply with 30% probability each
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    if rng.random() < 0.2:
        # Very slight rotation (± 1°)
        angle = rng.uniform(-1.0, 1.0)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=bg_color)

    gt_text = "\n".join(ground_truth)
    return img, gt_text


def run(
    n: int,
    raw_dir: Path,
    out_jsonl: Path,
    seed: int,
    probe_only: bool,
    probe_out: Path | None,
) -> int:
    rng = random.Random(seed)
    font_path = _ensure_font()

    corpus = _load_hindi_corpus(min_lines=max(500, n))
    print(f"Corpus: {len(corpus)} Hindi lines")

    if probe_only:
        probe_out = probe_out or Path("/tmp/synth_devanagari_probe")
        probe_out.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img, gt = _render_page(corpus, font_path, rng)
            img_path = probe_out / f"probe_{i:02d}.png"
            gt_path = probe_out / f"probe_{i:02d}.txt"
            img.save(img_path, "PNG")
            gt_path.write_text(gt, encoding="utf-8")
            print(f"  wrote {img_path} + {gt_path.name}")
        return 3

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    pages_dir = raw_dir / "devanagari_synthetic" / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in range(n):
            img, gt = _render_page(corpus, font_path, rng)
            img_path = pages_dir / f"{i:06d}.png"
            gt_path = pages_dir / f"{i:06d}.txt"
            img.save(img_path, "PNG")
            gt_path.write_text(gt, encoding="utf-8")

            rel = str(img_path.relative_to(raw_dir))
            f.write(json.dumps({
                "image":  rel,
                "text":   gt,
                "source": "devanagari_synthetic",
            }, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{n} pages rendered", flush=True)

    print(f"devanagari_synthetic: {written:,} pages → {out_jsonl}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=2000, help="Number of pages to render")
    p.add_argument("--raw_dir",   default="data/raw")
    p.add_argument("--out_jsonl", default="data/interim/devanagari_synthetic.jsonl")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run_probe", action="store_true",
                   help="Render 3 sample pages to --out (default /tmp/synth_devanagari_probe/), no JSONL")
    p.add_argument("--out", type=str, default="", help="Probe output directory")
    args = p.parse_args()

    probe_out = Path(args.out) if args.out else None
    run(
        n=args.n,
        raw_dir=Path(args.raw_dir),
        out_jsonl=Path(args.out_jsonl),
        seed=args.seed,
        probe_only=args.dry_run_probe,
        probe_out=probe_out,
    )


if __name__ == "__main__":
    main()
