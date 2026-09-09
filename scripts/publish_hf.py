"""Publish a LoRA adapter to the HuggingFace Hub.

Iter-3 and iter-4 were published manually. This script standardises the
step for iter-5+ so it becomes one command in the runbook.

Usage:
    export HF_TOKEN=hf_...   # write-scope token
    python scripts/publish_hf.py \\
        --adapter_dir outputs/iter5 \\
        --repo_id     enclavelabs/enclave-scribe-devanagari-iter5 \\
        --model_card  reports/iter5/MODEL_CARD.md

The script:
  1. Creates the repo if it doesn't exist (idempotent, exist_ok=True)
  2. Uploads the adapter folder (adapter_config.json, adapter weights,
     tokenizer files if present) via HfApi.upload_folder
  3. Uploads the model card as README.md (HF renders it on the model page)
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Publish a LoRA adapter to HuggingFace")
    parser.add_argument("--adapter_dir", required=True, help="Local path to adapter folder (e.g. outputs/iter5)")
    parser.add_argument("--repo_id",     required=True, help="HF repo id, e.g. enclavelabs/enclave-scribe-devanagari-iter5")
    parser.add_argument("--model_card",  default="",    help="Optional path to a MODEL_CARD.md; uploaded as README.md")
    parser.add_argument("--token",       default="",    help="HF write token; defaults to $HF_TOKEN")
    parser.add_argument("--private",     action="store_true", help="Create repo as private (default public)")
    parser.add_argument("--commit_msg",  default="",    help="Commit message for the upload")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_dir():
        sys.exit(f"error: --adapter_dir does not exist or is not a directory: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        sys.exit(f"error: {adapter_dir}/adapter_config.json missing — this doesn't look like a PEFT adapter folder")

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        sys.exit("error: no HF token found (pass --token or set HF_TOKEN)")

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"repo ready: https://huggingface.co/{args.repo_id}")

    commit_msg = args.commit_msg or f"Publish adapter from {adapter_dir.name}"
    api.upload_folder(
        folder_path=str(adapter_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=commit_msg,
    )
    print(f"uploaded adapter folder: {adapter_dir}")

    if args.model_card:
        card_path = Path(args.model_card)
        if not card_path.is_file():
            sys.exit(f"error: --model_card path does not exist: {card_path}")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=f"Publish model card from {card_path.name}",
        )
        print(f"uploaded model card as README.md from {card_path}")

    print(f"\ndone → https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
