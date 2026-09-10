"""Streamlit review tool for iter-6 human-label pilot.

Loads a pre-labeled JSONL (one iter-3 transcript per Devanagari page),
shows image + editable text side by side, and writes reviewer-corrected
labels to an output JSONL. Human reviewer's job: correct every error in
the iter-3 draft; do NOT re-transcribe from scratch.

Input JSONL format (one per line):
    {"image": "indicdlp_pages/hi/0001.jpg", "text": "<iter-3 pre-label>"}

Output JSONL format (one per line):
    {"image": "...", "text": "<reviewer-corrected>",
     "iter3_original": "<pre-label>", "reviewer": "...",
     "reviewed_at": "2026-09-11T12:34:56Z"}

Usage:
    pip install -e ".[label]"
    streamlit run scripts/label/review.py -- \\
        --input   data/interim/iter6_prelabels.jsonl \\
        --output  data/interim/iter6_labels.jsonl \\
        --image_root data/raw \\
        --reviewer "shashank"

CLI args go after `--` so Streamlit doesn't try to parse them.
"""
import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to iter-3 pre-labels JSONL")
    p.add_argument("--output", required=True, help="Path to write corrected labels JSONL")
    p.add_argument("--image_root", default="data/raw", help="Root for relative image paths")
    p.add_argument("--reviewer", default=os.environ.get("USER", "unknown"),
                   help="Reviewer identifier; also settable via $USER")
    # streamlit swallows argv up to `--`; anything after is ours.
    argv = sys.argv[1:]
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    return p.parse_args(argv)


@st.cache_data(show_spinner=False)
def _load_prelabels(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _load_reviewed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    seen = set()
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        seen.add(json.loads(line)["image"])
    return seen


def _append_reviewed(path: str, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()

    st.set_page_config(page_title="EnclaveScribe — iter-6 review", layout="wide")
    st.title("iter-6 Devanagari label review")
    st.caption(
        f"Reviewer: **{args.reviewer}** · Input: `{args.input}` · Output: `{args.output}`"
    )

    if not os.path.exists(args.input):
        st.error(f"Input file not found: {args.input}")
        st.stop()

    prelabels = _load_prelabels(args.input)
    reviewed = _load_reviewed_ids(args.output)
    todo = [r for r in prelabels if r["image"] not in reviewed]

    st.progress(len(reviewed) / max(len(prelabels), 1),
                text=f"{len(reviewed)}/{len(prelabels)} reviewed")

    if not todo:
        st.success("All pages reviewed.")
        st.stop()

    # Session-state cursor into the remaining todo list; reset when the
    # underlying file changes size (e.g. someone re-ran prelabeling).
    if st.session_state.get("_todo_len") != len(todo):
        st.session_state["_idx"] = 0
        st.session_state["_todo_len"] = len(todo)

    idx = st.session_state["_idx"]
    idx = max(0, min(idx, len(todo) - 1))
    record = todo[idx]

    st.subheader(f"Page {idx + 1} of {len(todo)}: `{record['image']}`")

    left, right = st.columns([1, 1], gap="medium")

    with left:
        image_path = os.path.join(args.image_root, record["image"])
        if os.path.exists(image_path):
            try:
                st.image(Image.open(image_path), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to open image: {e}")
        else:
            st.warning(f"Image not found at {image_path}")

    with right:
        st.markdown("**Iter-3 pre-label (read-only, for reference):**")
        st.code(record["text"], language="text")

        st.markdown("**Corrected text (edit below):**")
        # Prefill with iter-3 output; reviewer just corrects errors.
        corrected = st.text_area(
            "corrected", value=record["text"], height=400,
            label_visibility="collapsed", key=f"corrected_{idx}",
        )

        col_back, col_skip, col_save = st.columns([1, 1, 2])

        with col_back:
            if st.button("← Prev", disabled=idx == 0):
                st.session_state["_idx"] = idx - 1
                st.rerun()

        with col_skip:
            if st.button("Skip →"):
                st.session_state["_idx"] = idx + 1
                st.rerun()

        with col_save:
            if st.button("Save + Next", type="primary"):
                out = {
                    "image": record["image"],
                    "text": corrected,
                    "iter3_original": record["text"],
                    "reviewer": args.reviewer,
                    "reviewed_at": dt.datetime.now(dt.timezone.utc)
                                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
                _append_reviewed(args.output, out)
                st.session_state["_idx"] = idx + 1
                # Clear cache so progress bar reflects the new write
                _load_prelabels.clear()
                st.rerun()


if __name__ == "__main__":
    main()
