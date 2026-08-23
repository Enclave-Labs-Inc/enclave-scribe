"""Precheck all upstream data sources before running the full prep pipeline.

Runs quickly (~1-2 min). For each dataset, verifies the endpoint responds
with useful data. Prints a go/no-go summary so you know which prep scripts
are worth running vs. which need upstream fixes first.

Usage:
    python scripts/prepare/precheck.py

Exit code: 0 if every source is reachable, 1 otherwise.
"""
import sys
import time
import traceback

import requests


TIMEOUT = 20


def _head(url: str, allow_redirects: bool = True):
    r = requests.head(url, timeout=TIMEOUT, allow_redirects=allow_redirects)
    return r.status_code, r.headers.get("content-length", "?")


def _try_hf(dataset: str, **kwargs) -> tuple[bool, str]:
    """Load 1 sample via HF streaming — proves auth + schema."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset, streaming=True, **kwargs)
        # Take the first available split
        split_key = next(iter(ds.keys())) if hasattr(ds, "keys") else None
        stream = ds[split_key] if split_key else ds
        sample = next(iter(stream))
        return True, f"OK, sample keys: {sorted(list(sample.keys()))[:4]}..."
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}"


def check_cord() -> tuple[bool, str]:
    return _try_hf("naver-clova-ix/cord-v2")


def check_funsd() -> tuple[bool, str]:
    return _try_hf("nielsr/funsd")


def check_omnidocbench() -> tuple[bool, str]:
    return _try_hf("opendatalab/OmniDocBench", trust_remote_code=True)


def check_idl() -> tuple[bool, str]:
    return _try_hf("cathyzheng/IDL-WDS", split="train", trust_remote_code=True)


def check_xfund() -> tuple[bool, str]:
    """XFUND zips live on GitHub releases. Test one language/split."""
    url = "https://github.com/doc-analysis/XFUND/releases/download/v1.0/zh.train.zip"
    try:
        code, size = _head(url)
        if code == 200:
            return True, f"HTTP {code}, size ~{size} bytes"
        return False, f"HTTP {code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_hiertext_annotations() -> tuple[bool, str]:
    url = "https://storage.googleapis.com/hiertext/hiertext/validation.jsonl.gz"
    try:
        code, size = _head(url)
        if code == 200:
            return True, f"HTTP {code}, size ~{size} bytes"
        return False, f"HTTP {code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_hiertext_images() -> tuple[bool, str]:
    """Test one known Open Images ID from HierText's validation set."""
    sample_id = "00901f3f309f11c8"  # arbitrary open-images id
    url = f"https://storage.googleapis.com/openimages/data/train/{sample_id}.jpg"
    try:
        code, _ = _head(url)
        if code == 200:
            return True, f"HTTP {code}"
        return False, f"HTTP {code} — Open Images URL pattern likely dead"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_textocr_annotations() -> tuple[bool, str]:
    url = "https://dl.fbaipublicfiles.com/textvqa/data/textocr/TextOCR_0.1_val.json"
    try:
        code, size = _head(url)
        if code == 200:
            return True, f"HTTP {code}, size ~{size} bytes"
        return False, f"HTTP {code}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_textocr_images() -> tuple[bool, str]:
    """TextOCR uses same Open Images URL pattern as HierText — same bug."""
    return check_hiertext_images()


CHECKS = [
    ("CORD (HF: naver-clova-ix/cord-v2)",               check_cord),
    ("FUNSD (HF: nielsr/funsd)",                        check_funsd),
    ("XFUND zips (GitHub releases)",                    check_xfund),
    ("HierText annotations (GCS)",                      check_hiertext_annotations),
    ("HierText images (Open Images CDN)",               check_hiertext_images),
    ("TextOCR annotations (fbaipublicfiles)",           check_textocr_annotations),
    ("TextOCR images (Open Images CDN)",                check_textocr_images),
    ("OmniDocBench (HF: opendatalab/OmniDocBench)",     check_omnidocbench),
    ("IDL-WDS (HF: cathyzheng/IDL-WDS, streaming)",     check_idl),
]


def main():
    print("=== EnclaveScribe data-source precheck ===\n")
    results = []
    t0 = time.time()

    for name, fn in CHECKS:
        t_start = time.time()
        try:
            ok, msg = fn()
        except Exception:
            ok, msg = False, f"UNCAUGHT: {traceback.format_exc().splitlines()[-1]}"
        dt = time.time() - t_start
        status = "PASS" if ok else "FAIL"
        print(f"[{status}]  {name:<50} ({dt:>4.1f}s)  {msg}")
        results.append((name, ok))

    total = time.time() - t0
    passed = sum(1 for _, ok in results if ok)
    print(f"\n{passed}/{len(results)} sources reachable ({total:.1f}s total)")

    if passed < len(results):
        print("\nFailed sources (fix these upstream URLs or swap datasets before iter-2):")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
