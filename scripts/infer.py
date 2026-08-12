"""Batch inference: image directory or PDF → markdown files."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def collect_images(image_dir: str) -> list[str]:
    files = []
    for root, _, names in os.walk(image_dir):
        for name in names:
            if name.lower().endswith(_EXTS):
                files.append(os.path.join(root, name))
    return sorted(files)


def run(args):
    from scribe.data.pdf import pdf_to_images
    from scribe.infer.local import infer_image
    from scribe.model.vlm import Qwen2VLModel

    model = Qwen2VLModel()
    model.load(args.model_dir)

    if args.pdf:
        image_paths = pdf_to_images(args.pdf)
        prefix = Path(args.pdf).stem
        jobs = [
            (p, Path(args.output_dir) / f"{prefix}_page_{i + 1:04d}.md")
            for i, p in enumerate(image_paths)
        ]
    else:
        image_paths = collect_images(args.image_dir)
        jobs = [
            (p, Path(args.output_dir) / (Path(p).stem + ".md"))
            for p in image_paths
        ]

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def process(item: tuple[str, Path]) -> int:
        image_path, out_path = item
        text = infer_image(model, image_path)
        out_path.write_text(text, encoding="utf-8")
        return len(text.split())

    total_tokens = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process, j): j for j in jobs}
        for future in tqdm(as_completed(futures), total=len(jobs)):
            total_tokens += future.result()

    print(f"Done — {len(jobs)} file(s), {total_tokens} tokens.")


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe batch inference")
    parser.add_argument("--image_dir", default="", help="Directory of images")
    parser.add_argument("--pdf", default="", help="PDF file to process page by page")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--model_dir", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
