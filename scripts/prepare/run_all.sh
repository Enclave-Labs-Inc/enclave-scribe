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

set -e

RAW_DIR=${RAW_DIR:-data/raw}
INTERIM_DIR=${INTERIM_DIR:-data/interim}
PROCESSED_DIR=${PROCESSED_DIR:-data/processed}

echo "=== EnclaveScribe Data Prep ==="
echo "Raw dir      : $RAW_DIR"
echo "Interim dir  : $INTERIM_DIR"
echo "Processed dir: $PROCESSED_DIR"
echo ""

mkdir -p "$RAW_DIR" "$INTERIM_DIR" "$PROCESSED_DIR"

echo "[1/7] CORD receipts..."
python scripts/prepare/prep_cord.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/cord.jsonl"

echo "[2/7] FUNSD forms..."
python scripts/prepare/prep_funsd.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/funsd.jsonl"

echo "[3/7] XFUND multilingual forms (7 languages)..."
python scripts/prepare/prep_xfund.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/xfund.jsonl"

echo "[4/7] HierText natural scene text..."
python scripts/prepare/prep_hiertext.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/hiertext.jsonl"

echo "[5/7] TextOCR natural images..."
python scripts/prepare/prep_textocr.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/textocr.jsonl"

echo "[6/7] IDL industry documents (250K sample)..."
python scripts/prepare/prep_idl.py \
    --raw_dir "$RAW_DIR" \
    --out_jsonl "$INTERIM_DIR/idl.jsonl" \
    --max_samples 250000

echo "[7/7] Merging all datasets..."
python scripts/prepare/merge.py \
    --interim_dir "$INTERIM_DIR" \
    --out_dir "$PROCESSED_DIR" \
    --val_ratio 0.02

echo ""
echo "=== Done. Ready to train: ==="
echo "  torchrun --nproc_per_node=2 scripts/train.py \\"
echo "    --config configs/train/lora_lambda_2xa100.yaml"
