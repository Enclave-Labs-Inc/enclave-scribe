"""S3 mirror infrastructure — iter-10's #1 investment.

Every iter-N corpus builder should call `read_or_mirror(dataset_id, limit)`
instead of hitting HuggingFace directly. That helper:
  1. Checks whether an S3 mirror already exists at
     `s3://enclave-scribe-checkpoints/data/mirrors/<sanitized_id>/data.jsonl`.
  2. If yes → streams the JSONL from S3 (fast, no HF rate limits).
  3. If no → falls through to `datasets.load_dataset(streaming=True)` with
     retry-with-backoff, uploads the collected records back to S3, and returns
     them.

WHY THIS EXISTS
    Iter-7a and iter-9 each burned $10-16 on HF rate limits (`us.aws.cdn.hf.co
    Read timed out`). The cost of mirroring is trivial (~$0.02/dataset on S3
    storage) and every future iter pulls for free.

USAGE (standalone)
    # Mirror one dataset (uploads to S3 if not already there)
    python scripts/prepare/mirror_datasets_to_s3.py \\
        --dataset allenai/olmOCR-mix-1025 --limit 15000

    # Dry-run: check S3 state, print plan, no writes
    python scripts/prepare/mirror_datasets_to_s3.py \\
        --dataset allenai/olmOCR-mix-1025 --dry_run --limit 100

USAGE (from prep scripts)
    from scripts.prepare.mirror_datasets_to_s3 import DatasetMirror

    mirror = DatasetMirror()
    for record in mirror.read_or_mirror("apoidea/pubtabnet-html", limit=5000):
        # record is a dict pulled from the mirror (or freshly streamed)
        process(record)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator


S3_BUCKET = "enclave-scribe-checkpoints"
S3_PREFIX = "data/mirrors"

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE = REPO_ROOT / "data" / "mirrors"


def _sanitize(dataset_id: str) -> str:
    """Turn 'org/name' into 'org__name' (safe for S3 keys and local dirs)."""
    return dataset_id.replace("/", "__").replace(":", "_")


class DatasetMirror:
    """Read a HF dataset from an S3 mirror if it exists, else stream + upload."""

    def __init__(self, bucket: str = S3_BUCKET, prefix: str = S3_PREFIX):
        self.bucket = bucket
        self.prefix = prefix
        self.local_cache = LOCAL_CACHE

    def s3_uri(self, dataset_id: str) -> str:
        return f"s3://{self.bucket}/{self.prefix}/{_sanitize(dataset_id)}/data.jsonl"

    def local_path(self, dataset_id: str) -> Path:
        return self.local_cache / _sanitize(dataset_id) / "data.jsonl"

    def check_s3(self, dataset_id: str) -> bool:
        """Return True if the mirror JSONL already exists on S3."""
        uri = self.s3_uri(dataset_id)
        r = subprocess.run(
            ["aws", "s3", "ls", uri],
            capture_output=True, text=True, check=False,
        )
        return r.returncode == 0 and uri.split("/")[-1] in r.stdout

    def _pull_from_s3(self, dataset_id: str) -> Path:
        """Download the mirror to local cache and return the local path."""
        local = self.local_path(dataset_id)
        local.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [mirror] pulling from S3: {self.s3_uri(dataset_id)}", file=sys.stderr)
        subprocess.run(
            ["aws", "s3", "cp", self.s3_uri(dataset_id), str(local)],
            check=True,
        )
        return local

    def _push_to_s3(self, dataset_id: str, local: Path) -> None:
        print(f"  [mirror] uploading to S3: {self.s3_uri(dataset_id)}", file=sys.stderr)
        subprocess.run(
            ["aws", "s3", "cp", str(local), self.s3_uri(dataset_id)],
            check=True,
        )

    def _stream_hf(self, dataset_id: str, limit: int, split: str = "train") -> Iterator[dict]:
        """Stream from HF with 3-retry exponential backoff (5, 15, 30s)."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError("`datasets` not installed. Run: pip install datasets") from e

        last_err = None
        for attempt, delay in enumerate([0, 5, 15, 30]):
            if delay:
                print(f"  [mirror] retry {attempt} after {delay}s ({last_err})", file=sys.stderr)
                time.sleep(delay)
            try:
                ds = load_dataset(dataset_id, split=split, streaming=True)
                yielded = 0
                for record in ds:
                    yield record
                    yielded += 1
                    if limit and yielded >= limit:
                        return
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"dataset unavailable after 3 retries: {dataset_id} — {last_err}")

    def mirror(self, dataset_id: str, limit: int = 0, split: str = "train") -> Path:
        """Force a fresh mirror pull from HF, upload to S3, return local path.

        Serializes each streamed record as a JSONL line. Images (PIL) are NOT
        embedded — the caller is responsible for image handling. For OCR datasets,
        the mirror stores metadata + reference paths; images can be re-streamed
        or fetched separately.
        """
        local = self.local_path(dataset_id)
        local.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(local, "w", encoding="utf-8") as f:
            for record in self._stream_hf(dataset_id, limit=limit, split=split):
                # Strip non-JSON-serializable fields (PIL images, etc.)
                cleaned = {}
                for k, v in record.items():
                    if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                        cleaned[k] = v
                    else:
                        cleaned[k] = f"<{type(v).__name__}>"
                f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                n += 1
        print(f"  [mirror] mirrored {n} records → {local}", file=sys.stderr)
        self._push_to_s3(dataset_id, local)
        return local

    def read_or_mirror(self, dataset_id: str, limit: int = 0, split: str = "train") -> Iterator[dict]:
        """Primary helper: yield records from mirror if it exists, else mirror+read."""
        if self.check_s3(dataset_id):
            local = self.local_path(dataset_id)
            if not local.exists():
                self._pull_from_s3(dataset_id)
            with open(local, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if limit and i >= limit:
                        break
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return

        # No mirror yet — create one, then re-read
        self.mirror(dataset_id, limit=limit, split=split)
        # Recurse via local file after upload
        local = self.local_path(dataset_id)
        with open(local, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                line = line.strip()
                if line:
                    yield json.loads(line)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True, help="HF dataset id, e.g. allenai/olmOCR-mix-1025")
    p.add_argument("--limit", type=int, default=0, help="Max samples to mirror (0 = full stream)")
    p.add_argument("--split", default="train")
    p.add_argument("--dry_run", action="store_true", help="Check S3 state, print plan, don't upload")
    p.add_argument("--force", action="store_true", help="Re-mirror even if S3 already has it")
    args = p.parse_args()

    m = DatasetMirror()
    exists = m.check_s3(args.dataset)
    print(f"dataset:  {args.dataset}")
    print(f"s3 uri:   {m.s3_uri(args.dataset)}")
    print(f"mirrored: {exists}")

    if args.dry_run:
        print("[dry-run] would mirror" if not exists else "[dry-run] already mirrored")
        return

    if exists and not args.force:
        print("already mirrored (use --force to re-mirror)")
        return

    m.mirror(args.dataset, limit=args.limit, split=args.split)
    print("done")


if __name__ == "__main__":
    main()
