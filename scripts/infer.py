"""Batch inference: any document type (image, PDF, PPT, DOCX) → markdown files."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from scribe.data.convert import SUPPORTED_EXTS


def collect_documents(doc_dir: str) -> list[str]:
    files = []
    for root, _, names in os.walk(doc_dir):
        for name in names:
            if Path(name).suffix.lower() in SUPPORTED_EXTS:
                files.append(os.path.join(root, name))
    return sorted(files)


def run(args):
    from scribe.infer.local import infer_document
    from scribe.model.vlm import Qwen2VLModel

    model = Qwen2VLModel()
    model.load(args.model_dir)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.file:
        file_paths = [args.file]
    else:
        file_paths = collect_documents(args.input_dir)

    def process(file_path: str) -> int:
        pages = infer_document(model, file_path, dpi=args.dpi)
        stem = Path(file_path).stem
        total = 0
        for i, text in enumerate(pages):
            suffix = f"_page_{i + 1:04d}" if len(pages) > 1 else ""
            out = Path(args.output_dir) / f"{stem}{suffix}.md"
            out.write_text(text, encoding="utf-8")
            total += len(text.split())
        return total

    total_tokens = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process, f): f for f in file_paths}
        for future in tqdm(as_completed(futures), total=len(file_paths)):
            total_tokens += future.result()

    print(f"Done — {len(file_paths)} file(s), {total_tokens} tokens.")


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe batch inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Single file (image, PDF, PPT, DOCX, …)")
    group.add_argument("--input_dir", help="Directory of documents (all supported types)")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--model_dir", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
