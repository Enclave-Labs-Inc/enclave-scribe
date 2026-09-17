#!/usr/bin/env bash
# Rehydrate himalaya_500 replay images into data/raw/ so training's
# image_root=data/raw resolves the JSONL's "himalaya_indic/<shard>/<idx>.jpg"
# paths. Iter-11 crashed at step 4 with FileNotFoundError because
# stage_devanagari_benchmark.py copies these under data/benchmark/, not
# data/raw/. Idempotent — safe to run on every launch.
set -euo pipefail

SRC="s3://enclave-scribe-checkpoints/data/benchmark/himalaya_500/himalaya_indic/"
DST="data/raw/himalaya_indic/"

existing=$(find "$DST" -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
if [ "$existing" -ge 500 ]; then
    echo "rehydrate_himalaya_replay: $DST already has $existing files, skipping sync"
    exit 0
fi

mkdir -p "$DST"
echo "rehydrate_himalaya_replay: syncing $SRC -> $DST"
aws s3 sync "$SRC" "$DST"
echo "rehydrate_himalaya_replay: done; $(find "$DST" -name '*.jpg' | wc -l | tr -d ' ') files"
