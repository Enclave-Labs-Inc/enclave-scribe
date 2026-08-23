#!/usr/bin/env bash
# Run all data prep scripts in sequence then merge.
# Usage: bash scripts/prepare/run_all.sh
# Estimated total download + prep time: 4-8 hours depending on network.
#
# Dataset sizes after prep:
#   CORD      ~11K samples    ~2 GB images
#   FUNSD       ~200 samples  ~50 MB
#   XFUND     ~1.4K samples   ~500 MB
#   HierText  ~11K samples    ~5 GB
#   TextOCR   ~28K samples    ~20 GB
#   IDL       250K samples    ~120 GB
#   ─────────────────────────────────
#   Total     ~300K samples   ~150 GB

# NB: no `set -e` — we want to continue if one dataset fails (broken URL,
# missing file, etc.). merge.py uses whatever .jsonl files exist.

RAW_DIR=${RAW_DIR:-data/raw}
INTERIM_DIR=${INTERIM_DIR:-data/interim}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed}
BENCHMARK_DIR=${BENCHMARK_DIR:-data/benchmark}
SKIP_LARGE=${SKIP_LARGE:-1}   # skip IDL (120GB) and HierText (broken URL) by default

echo "=== EnclaveScribe Data Prep ==="
echo "Raw dir      : $RAW_DIR"
echo "Interim dir  : $INTERIM_DIR"
echo "Processed dir: $PROCESSED_DIR"
echo "Benchmark dir: $BENCHMARK_DIR (official test splits — never used in training)"
echo "Skip large   : $SKIP_LARGE (set SKIP_LARGE=0 to include IDL + HierText)"
echo ""

mkdir -p "$RAW_DIR" "$INTERIM_DIR" "$PROCESSED_DIR" "$BENCHMARK_DIR"

run_step() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    if "$@"; then
        echo "$name: OK"
    else
        echo "$name: FAILED (skipping — pipeline continues)"
    fi
}

run_step "[1/7] CORD receipts" python scripts/prepare/prep_cord.py \
    --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/cord.jsonl" \
    --benchmark_jsonl "$BENCHMARK_DIR/cord_test.jsonl"

run_step "[2/7] FUNSD forms" python scripts/prepare/prep_funsd.py \
    --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/funsd.jsonl" \
    --benchmark_jsonl "$BENCHMARK_DIR/funsd_test.jsonl"

run_step "[3/7] XFUND multilingual forms" python scripts/prepare/prep_xfund.py \
    --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/xfund.jsonl"

if [ "$SKIP_LARGE" = "0" ]; then
    run_step "[4/7] HierText natural scene text" python scripts/prepare/prep_hiertext.py \
        --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/hiertext.jsonl"
else
    echo ""
    echo "[4/7] HierText: SKIPPED (broken upstream URL — set SKIP_LARGE=0 to retry)"
fi

run_step "[5/7] TextOCR natural images" python scripts/prepare/prep_textocr.py \
    --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/textocr.jsonl"

if [ "$SKIP_LARGE" = "0" ]; then
    run_step "[6/7] IDL industry documents (250K sample)" python scripts/prepare/prep_idl.py \
        --raw_dir "$RAW_DIR" --out_jsonl "$INTERIM_DIR/idl.jsonl" --max_samples 250000
else
    echo ""
    echo "[6/7] IDL: SKIPPED (~120GB download — set SKIP_LARGE=0 to include)"
fi

echo ""
echo "=== [7/8] Merging all datasets that succeeded ==="
python scripts/prepare/merge.py \
    --interim_dir "$INTERIM_DIR" \
    --out_dir "$PROCESSED_DIR" \
    --val_ratio 0.02

echo ""
echo "=== [8/8] Aggregating held-out test splits ==="
HELDOUT="$BENCHMARK_DIR/heldout_test.jsonl"
: > "$HELDOUT"
shopt -s nullglob
for f in "$BENCHMARK_DIR"/*_test.jsonl; do
    [ "$f" = "$HELDOUT" ] && continue
    cat "$f" >> "$HELDOUT"
done
n=$(wc -l < "$HELDOUT" | tr -d ' ')
echo "Held-out test set: $n samples → $HELDOUT"

echo ""
echo "=== Done. Ready to train: ==="
echo "  torchrun --nproc_per_node=2 scripts/train.py \\"
echo "    --config configs/train/lora_lambda_2xa100.yaml"
echo ""
echo "After training, evaluate on the held-out test set:"
echo "  python scripts/eval.py \\"
echo "    --gt_jsonl   $HELDOUT \\"
echo "    --image_root $RAW_DIR \\"
echo "    --base_model Qwen/Qwen2.5-VL-7B-Instruct \\"
echo "    --adapter_dir outputs/iter1 \\"
echo "    --out_json   results/iter1_heldout.json"
