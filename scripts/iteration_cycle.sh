#!/usr/bin/env bash
# Full iteration cycle: download weights → predict on train → label-rank → report
# Usage: ./scripts/iteration_cycle.sh <weights_path> <cycle_number>
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source 3lc-env/bin/activate

WEIGHTS="${1:-kaggle_overnight_output/best.pt}"
CYCLE="${2:-2}"

echo "================================================"
echo "  ITERATION CYCLE $CYCLE"
echo "  Weights: $WEIGHTS"
echo "================================================"

# 1. Predict on training set (CPU, parallel-safe)
echo ""
echo "[1/4] Predicting on training set..."
rm -rf predictions_train
python scripts/predict_on_train.py \
    --weights "$WEIGHTS" \
    --split train \
    --device cpu \
    --conf 0.1 \
    --batch 16

# 2. Predict on val set
echo ""
echo "[2/4] Predicting on validation set..."
rm -rf predictions_val
python scripts/predict_on_train.py \
    --weights "$WEIGHTS" \
    --split val \
    --device cpu \
    --conf 0.1 \
    --batch 16 \
    --output predictions_val

# 3. Run label-ranker (Rust, instant)
echo ""
echo "[3/4] Running label-ranker..."
RANKER="$REPO_ROOT/rust/target/release/label-ranker"
EVAL="$REPO_ROOT/rust/target/release/map-eval"

"$RANKER" \
    --predictions predictions_train/ \
    --labels data/train/labels/ \
    --top 100 > label_issues_cycle${CYCLE}.json

echo "Top 20 issues:"
"$RANKER" \
    --predictions predictions_train/ \
    --labels data/train/labels/ \
    --top 20 \
    --format tsv

# 4. Local mAP eval
echo ""
echo "[4/4] Local mAP@0.5 on validation..."
"$EVAL" --predictions predictions_val/ --labels data/val/labels/

echo ""
echo "================================================"
echo "  CYCLE $CYCLE ANALYSIS COMPLETE"
echo "  Issues: label_issues_cycle${CYCLE}.json"
echo "  Next: review issues with label-fixer agent"
echo "================================================"
