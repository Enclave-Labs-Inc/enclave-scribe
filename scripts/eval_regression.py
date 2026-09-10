"""EnclaveScribe cross-adapter regression harness.

Runs any combination of three tests against N adapters and emits a
markdown comparison table. Any test can be skipped by omitting its
input flag.

  1. Devanagari word CER   (--devanagari_jsonl PATH)
  2. English heldout CER   (--english_jsonl PATH)
  3. Gazette page gate     (--gazette_pdf PATH)

The harness shells out to scripts/eval.py and scripts/agent/parse.py so
each run gets a fresh CUDA process (releases VRAM cleanly between
adapters; important for 32B where two live models don't co-fit).

For the gazette gate, `bad_words_ids` in scribe/agent/tools.py must be
DISABLED so 0 `<tool_call>` blocks actually proves the model no longer
loops on long pages. See tests/fixtures/pdfs/README.md for the five
pass criteria.

Usage:
    python scripts/eval_regression.py \\
        --base_model       allenai/olmOCR-2-7B-1025 \\
        --adapter          base: \\
        --adapter          iter3:outputs/iter3 \\
        --adapter          iter4:outputs/iter4 \\
        --devanagari_jsonl data/benchmark/iter3_words_500.jsonl \\
        --english_jsonl    data/benchmark/heldout_500.jsonl \\
        --gazette_pdf      tests/fixtures/pdfs/gazette_moef_2024_06_07.pdf \\
        --out              results/iter5/regression_table.md \\
        --work_dir         results/iter5/regression
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ITER4_GAZETTE_BASELINE_CHARS = 13646
GAZETTE_MIN_CHARS = 12000
GAZETTE_COHERENT_MIN_DEVANAGARI = 100
GAZETTE_COHERENT_MAX_REPEAT_RATIO = 0.30
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def parse_adapter_arg(s: str) -> tuple[str, str]:
    """`name:path` → (name, path). Empty path means base model only."""
    if ":" not in s:
        sys.exit(f"error: --adapter must be name:path, got {s!r}")
    name, path = s.split(":", 1)
    if not name:
        sys.exit(f"error: --adapter name cannot be empty, got {s!r}")
    return name, path


def run_eval_jsonl(adapter_path: str, base_model: str, gt_jsonl: str,
                   image_root: str, out_json: Path) -> dict:
    """Invoke scripts/eval.py; return the parsed JSON payload."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/eval.py",
        "--gt_jsonl",   gt_jsonl,
        "--image_root", image_root,
        "--base_model", base_model,
        "--adapter_dir", adapter_path,
        "--out_json",   str(out_json),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return json.loads(out_json.read_text())


def run_gazette(adapter_path: str, base_model: str, pdf: str,
                out_md: Path, report_json: Path) -> dict:
    """Invoke scripts/agent/parse.py and compute gate metrics from the output."""
    out_md.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/agent/parse.py",
        "--pdf",         pdf,
        "--out",         str(out_md),
        "--base_model",  base_model,
        "--adapter_dir", adapter_path,
        "--report_json", str(report_json),
    ]
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return gazette_gate_metrics(out_md, report_json)


def gazette_gate_metrics(out_md: Path, report_json: Path) -> dict:
    text = out_md.read_text(encoding="utf-8")
    total_chars = len(text)
    tool_call_count = text.count("<tool_call>")
    devanagari_chars = len(DEVANAGARI_RE.findall(text))

    lines = [ln for ln in text.splitlines() if ln.strip()]
    repeat_ratio = 0.0
    if lines:
        from collections import Counter
        top = Counter(lines).most_common(1)[0][1]
        repeat_ratio = top / len(lines)

    is_coherent = (
        devanagari_chars >= GAZETTE_COHERENT_MIN_DEVANAGARI
        and repeat_ratio < GAZETTE_COHERENT_MAX_REPEAT_RATIO
    )

    report = json.loads(report_json.read_text()) if report_json.exists() else {}
    n_pages = report.get("n_pages", 0)

    gate_1_no_tool_calls  = tool_call_count == 0
    gate_3_min_chars      = total_chars >= GAZETTE_MIN_CHARS
    gate_4_coherent       = is_coherent
    gate_5_beats_iter4    = total_chars >= ITER4_GAZETTE_BASELINE_CHARS
    all_gates_pass = all([gate_1_no_tool_calls, gate_3_min_chars,
                          gate_4_coherent, gate_5_beats_iter4])

    return {
        "total_chars":       total_chars,
        "devanagari_chars":  devanagari_chars,
        "tool_call_count":   tool_call_count,
        "n_pages":           n_pages,
        "repeat_ratio":      round(repeat_ratio, 3),
        "is_coherent":       is_coherent,
        "gate_1_no_tool_calls": gate_1_no_tool_calls,
        "gate_3_min_chars":     gate_3_min_chars,
        "gate_4_coherent":      gate_4_coherent,
        "gate_5_beats_iter4":   gate_5_beats_iter4,
        "all_gates_pass":       all_gates_pass,
    }


def emit_table(rows: list[dict], out_md: Path) -> None:
    """Emit a markdown comparison table across adapters."""
    lines = []
    lines.append("# EnclaveScribe regression comparison\n")
    lines.append("Head-to-head across adapters. `-` means the test was not run for that adapter.\n")

    have_dev  = any(r.get("devanagari_cer") is not None for r in rows)
    have_eng  = any(r.get("english_cer")    is not None for r in rows)
    have_gaz  = any(r.get("gazette")        is not None for r in rows)

    headers = ["Adapter"]
    if have_dev:
        headers += ["Devanagari CER↓", "N"]
    if have_eng:
        headers += ["English CER↓", "N"]
    if have_gaz:
        headers += ["Gazette chars", "tool_call", "pages", "gate pass"]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for r in rows:
        row = [r["name"]]
        if have_dev:
            d = r.get("devanagari_cer")
            row += [f"{d['cer']:.4f}" if d else "-", str(d["n"]) if d else "-"]
        if have_eng:
            e = r.get("english_cer")
            row += [f"{e['cer']:.4f}" if e else "-", str(e["n"]) if e else "-"]
        if have_gaz:
            g = r.get("gazette")
            if g:
                row += [str(g["total_chars"]), str(g["tool_call_count"]),
                        str(g["n_pages"]), "✅" if g["all_gates_pass"] else "❌"]
            else:
                row += ["-", "-", "-", "-"]
        lines.append("| " + " | ".join(row) + " |")

    if have_gaz:
        lines.append("")
        lines.append("**Gazette gate criteria** (from tests/fixtures/pdfs/README.md):")
        lines.append(f"1. `<tool_call>` count = 0")
        lines.append(f"3. Total chars ≥ {GAZETTE_MIN_CHARS}")
        lines.append(f"4. Coherent = ≥ {GAZETTE_COHERENT_MIN_DEVANAGARI} Devanagari chars AND top-line repeat ratio < {GAZETTE_COHERENT_MAX_REPEAT_RATIO}")
        lines.append(f"5. Total chars ≥ iter-4 baseline ({ITER4_GAZETTE_BASELINE_CHARS})")
        lines.append("")
        lines.append("Criterion 2 (no page hit max_new_tokens) is qualitative — inspect the per-adapter report_json for pages with anomalously short markdown alongside a full-token budget.")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote regression table → {out_md}")


def main():
    parser = argparse.ArgumentParser(description="EnclaveScribe cross-adapter regression harness")
    parser.add_argument("--adapter",          action="append", required=True,
                        help="Repeatable: name:path (empty path = base model only). E.g. iter4:outputs/iter4")
    parser.add_argument("--base_model",       required=True, help="Base model id (must match all adapters)")
    parser.add_argument("--image_root",       default="data/raw")
    parser.add_argument("--devanagari_jsonl", default="", help="Devanagari word CER JSONL; skip test if empty")
    parser.add_argument("--english_jsonl",    default="", help="English heldout CER JSONL; skip test if empty")
    parser.add_argument("--gazette_pdf",      default="", help="Gazette page-level PDF; skip gate if empty")
    parser.add_argument("--out",              default="results/regression/regression_table.md")
    parser.add_argument("--work_dir",         default="results/regression",
                        help="Where per-adapter JSON/MD outputs land")
    args = parser.parse_args()

    if not any([args.devanagari_jsonl, args.english_jsonl, args.gazette_pdf]):
        sys.exit("error: pass at least one of --devanagari_jsonl, --english_jsonl, --gazette_pdf")

    adapters = [parse_adapter_arg(s) for s in args.adapter]
    work = Path(args.work_dir)

    rows = []
    for name, path in adapters:
        print(f"\n{'='*60}\nAdapter: {name}  path={path or '(base)'}\n{'='*60}")
        row = {"name": name}

        if args.devanagari_jsonl:
            out_json = work / f"{name}_devanagari.json"
            r = run_eval_jsonl(path, args.base_model, args.devanagari_jsonl,
                               args.image_root, out_json)
            row["devanagari_cer"] = {"cer": r["overall"]["cer"], "n": r["n_samples"]}

        if args.english_jsonl:
            out_json = work / f"{name}_english.json"
            r = run_eval_jsonl(path, args.base_model, args.english_jsonl,
                               args.image_root, out_json)
            row["english_cer"] = {"cer": r["overall"]["cer"], "n": r["n_samples"]}

        if args.gazette_pdf:
            out_md      = work / f"{name}_gazette.md"
            report_json = work / f"{name}_gazette_report.json"
            row["gazette"] = run_gazette(path, args.base_model, args.gazette_pdf,
                                         out_md, report_json)

        rows.append(row)

    emit_table(rows, Path(args.out))


if __name__ == "__main__":
    main()
