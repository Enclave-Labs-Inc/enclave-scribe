#!/usr/bin/env python3
"""Local smoke test for the prep_*.py scripts.

Catches the silent-fail class of bug that shipped iter-11's under-sized
corpus. For each prep it runs the script with a tiny limit against a temp
JSONL and asserts the output is non-empty and shaped correctly.

Runs in <2 min on a laptop (no GPU, small HF pulls). Invoked manually as
step in the iter-12 launch checklist — no CI wiring.

Skipped: prep_olmocr_mix.py, whose smoke path needs a multi-GB PDF tarball
download. Verify that one manually on the AWS instance.

USAGE
    python scripts/prepare/smoke_test_preps.py [--limit 3] [--out_dir /tmp/smoke_preps]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PrepSpec:
    name: str
    script: str          # relative to REPO_ROOT
    limit_flag: str      # e.g. "--limit" or "--max_samples"


PREPS: list[PrepSpec] = [
    PrepSpec("chartqa",      "scripts/prepare/prep_chartqa.py",      "--max_samples"),
    PrepSpec("math_formula", "scripts/prepare/prep_math_formula.py", "--limit"),
    PrepSpec("pubtabnet",    "scripts/prepare/prep_pubtabnet.py",    "--limit"),
    # olmocr_mix intentionally skipped — its smoke path renders PDFs from
    # multi-GB tarballs, not a laptop-scale check. Verify manually.
]

TIMEOUT_SECS = 300  # allows ~1GB ChartQA parquet pull over consumer bandwidth


@dataclass
class Result:
    name: str
    status: str          # OK | FAIL | SKIP
    samples: int
    duration_s: float
    note: str = ""


def _validate_jsonl(path: Path) -> tuple[bool, int, str]:
    """Return (ok, sample_count, reason). Reads only the first line."""
    if not path.exists():
        return False, 0, "output file missing"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return False, 0, "output empty"
    try:
        row = json.loads(lines[0])
    except json.JSONDecodeError as e:
        return False, len(lines), f"first line not JSON: {e}"
    for key in ("image", "text"):
        val = row.get(key)
        if not isinstance(val, str) or not val:
            return False, len(lines), f"first row missing/empty {key!r}"
    return True, len(lines), ""


def _run_one(spec: PrepSpec, limit: int, out_dir: Path) -> Result:
    out_jsonl = out_dir / f"smoke_{spec.name}.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if out_jsonl.exists():
        out_jsonl.unlink()

    cmd = [
        sys.executable,
        spec.script,
        spec.limit_flag, str(limit),
        "--out_jsonl", str(out_jsonl),
    ]
    print(f"\n>>> {spec.name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    timed_out = False
    proc_returncode = 0
    proc_stderr = ""
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            timeout=TIMEOUT_SECS,
            capture_output=True,
            text=True,
        )
        proc_returncode = proc.returncode
        proc_stderr = proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        proc_stderr = (e.stderr or b"").decode("utf-8", errors="replace") if e.stderr else ""
    duration = time.time() - t0

    # Validate the JSONL regardless of exit path: some HF-backed preps produce
    # their samples quickly then linger on interpreter shutdown (background
    # threads in `datasets`/`urllib3`). Non-zero exit / timeout is only a real
    # failure if the output JSONL is also missing/invalid.
    ok, n, reason = _validate_jsonl(out_jsonl)

    if ok:
        note = "slow exit (samples produced)" if (timed_out or proc_returncode != 0) else ""
        return Result(spec.name, "OK", n, duration, note)

    if timed_out:
        return Result(spec.name, "FAIL", n, duration, f"timeout ({reason})")
    if proc_returncode != 0:
        tail = proc_stderr.strip().splitlines()[-1:] or ["(no stderr)"]
        return Result(spec.name, "FAIL", n, duration, f"exit {proc_returncode}: {tail[-1]}")
    return Result(spec.name, "FAIL", n, duration, reason)


def _print_summary(results: list[Result]) -> None:
    print("\nSmoke test results:")
    print(f"  {'source':<16} {'status':<7} {'samples':>8}  {'duration':>10}  note")
    for r in results:
        samples = "-" if r.status == "SKIP" else str(r.samples)
        duration = "-" if r.status == "SKIP" else f"{r.duration_s:.1f}s"
        note = f"  {r.note}" if r.note else ""
        print(f"  {r.name:<16} {r.status:<7} {samples:>8}  {duration:>10}{note}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--limit",   type=int, default=3, help="Samples per prep")
    p.add_argument("--out_dir", default="/tmp/smoke_preps",
                   help="Directory for temp smoke JSONL outputs")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    results: list[Result] = []
    for spec in PREPS:
        results.append(_run_one(spec, args.limit, out_dir))

    results.append(Result(
        "olmocr_mix", "SKIP", 0, 0.0,
        "heavy PDF render; verify manually",
    ))

    _print_summary(results)

    any_failed = any(r.status == "FAIL" for r in results)
    return 2 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
