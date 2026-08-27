"""himalaya-ai/devanagari_ocr_dataset — real Indic OCR data curated from 7 sources.

Dataset : himalaya-ai/devanagari_ocr_dataset  (~58 GB, ShareGPT format)
Content : Hindi, Marathi, Sanskrit, Nepali, Pali document + word OCR
Sources : IIT-Indic-HW-Words, IndicVisionBench, NayanaBench, and more
Output  : data/raw/himalaya_indic/<shard>/<idx>.jpg
          data/interim/himalaya_indic.jsonl
          data/benchmark/himalaya_indic_test.jsonl  (2% held out)

The upstream schema is ShareGPT-style:
    {
      "image":         "images/xxx.jpg",
      "conversations": [
        {"from": "human", "value": "<image>\\nText Recognition:"},
        {"from": "gpt",   "value": "स्वर्ग आफै"}
      ]
    }

We convert to our DocumentDataset schema:
    {
      "image":  "himalaya_indic/<shard>/<idx>.jpg",
      "text":   "स्वर्ग आफै",
      "prompt": "Text Recognition:"          # optional per-sample prompt
    }

Filtering rules:
- Skip samples with empty text
- Skip samples where image can't be opened
- Skip samples with text < MIN_TEXT_CHARS (default 3)
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


DATASET_ID    = "himalaya-ai/devanagari_ocr_dataset"
SHARD_SIZE    = 5000
MIN_TEXT_CHARS = 3


def _extract_conversation_text(conversations) -> tuple[str, str]:
    """Pull (human_prompt, gpt_answer) out of the ShareGPT conversation list.

    Human values may contain "<image>\\n" prefix; we strip that so the prompt
    is just the task instruction. gpt answer becomes the training target.
    """
    human = ""
    gpt   = ""
    for turn in (conversations or []):
        who = str(turn.get("from", "")).lower()
        val = str(turn.get("value", "")).strip()
        if who == "human":
            human = val.replace("<image>", "").strip()
        elif who in ("gpt", "assistant"):
            gpt = val
    return human, gpt


def run(
    raw_dir: Path,
    out_jsonl: Path,
    benchmark_jsonl: Path | None,
    max_samples: int = 100_000,
    val_ratio: float = 0.02,
    seed: int = 42,
) -> int:
    import random
    rng = random.Random(seed)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if benchmark_jsonl is not None:
        benchmark_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_ID} (target {max_samples:,} samples)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    img_base = raw_dir / "himalaya_indic"
    img_base.mkdir(parents=True, exist_ok=True)

    train_count = 0
    test_count  = 0
    skipped     = 0
    shard       = 0
    shard_dir   = img_base / f"{shard:04d}"
    shard_dir.mkdir(exist_ok=True)

    train_f = open(out_jsonl, "w", encoding="utf-8")
    test_f  = open(benchmark_jsonl, "w", encoding="utf-8") if benchmark_jsonl else None

    try:
        pbar = tqdm(total=max_samples, desc="himalaya-ai")
        for sample in ds:
            if train_count + test_count >= max_samples:
                break

            # Extract prompt + target text from the ShareGPT conversation.
            human_prompt, gpt_answer = _extract_conversation_text(
                sample.get("conversations", [])
            )
            if not gpt_answer or len(gpt_answer.strip()) < MIN_TEXT_CHARS:
                skipped += 1
                continue

            # Get the image — HF returns a PIL Image when the field is decoded.
            img = sample.get("image")
            if img is None:
                skipped += 1
                continue
            try:
                if not hasattr(img, "convert"):
                    # Sometimes the image arrives as dict {"bytes": ..., "path": ...}
                    if isinstance(img, dict) and "bytes" in img:
                        import io
                        img = Image.open(io.BytesIO(img["bytes"]))
                    else:
                        skipped += 1
                        continue
                img = img.convert("RGB")
            except Exception:
                skipped += 1
                continue

            # Route to train vs benchmark by ratio (only when benchmark file open)
            is_test = (test_f is not None and rng.random() < val_ratio)
            target_f = test_f if is_test else train_f
            counter  = "test_count" if is_test else "train_count"

            # Rotate shard every SHARD_SIZE samples (across both train+test)
            total = train_count + test_count
            if total > 0 and total % SHARD_SIZE == 0:
                shard += 1
                shard_dir = img_base / f"{shard:04d}"
                shard_dir.mkdir(exist_ok=True)

            # Save image
            fname = f"{total % SHARD_SIZE:05d}.jpg"
            img_path = shard_dir / fname
            img.save(img_path, "JPEG", quality=90)

            rel = str(img_path.relative_to(raw_dir))
            record = {"image": rel, "text": gpt_answer}
            if human_prompt:
                record["prompt"] = human_prompt
            if is_test:
                record["category"] = "himalaya_indic"

            target_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if is_test:
                test_count += 1
            else:
                train_count += 1
                pbar.update(1)

        pbar.close()
    finally:
        train_f.close()
        if test_f is not None:
            test_f.close()

    print(f"himalaya-ai: {train_count:,} train → {out_jsonl}")
    if benchmark_jsonl:
        print(f"himalaya-ai: {test_count:,} held out → {benchmark_jsonl}")
    print(f"skipped: {skipped:,} (empty text / broken image / short text)")
    return train_count + test_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",         default="data/raw")
    parser.add_argument("--out_jsonl",       default="data/interim/himalaya_indic.jsonl")
    parser.add_argument("--benchmark_jsonl", default="",
                        help="If set, val_ratio of samples are routed here as held-out test")
    parser.add_argument("--max_samples",     type=int, default=100_000,
                        help="Cap on total samples emitted (train + held-out)")
    parser.add_argument("--val_ratio",       type=float, default=0.02,
                        help="Fraction of samples routed to benchmark_jsonl")
    args = parser.parse_args()
    bench = Path(args.benchmark_jsonl) if args.benchmark_jsonl else None
    run(
        Path(args.raw_dir),
        Path(args.out_jsonl),
        bench,
        max_samples=args.max_samples,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
