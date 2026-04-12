#!/usr/bin/env bash
# Ultra-fast iteration script for 3LC competition.
# Usage: ./scripts/iterate.sh [--skip-train] [--skip-predict] [--top N]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_TRAIN=false
SKIP_PREDICT=false
TOP=50

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train) SKIP_TRAIN=true; shift ;;
        --skip-predict) SKIP_PREDICT=true; shift ;;
        --top) TOP="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

RANKER="$REPO_ROOT/rust/target/release/label-ranker"
EVAL="$REPO_ROOT/rust/target/release/map-eval"

# Check binaries exist
if [[ ! -f "$RANKER" ]] || [[ ! -f "$EVAL" ]]; then
    echo "Building Rust tools..."
    (cd rust && cargo build --release)
fi

echo "================================================"
echo "  3LC ITERATION CYCLE"
echo "================================================"

# Step 1: Train (if not skipping)
if [[ "$SKIP_TRAIN" == false ]]; then
    echo ""
    echo "[1/5] Training YOLOv8n..."
    python train.py
else
    echo ""
    echo "[1/5] Skipping training (--skip-train)"
fi

# Step 2: Predict on training set
if [[ "$SKIP_PREDICT" == false ]]; then
    echo ""
    echo "[2/5] Predicting on training set..."
    python scripts/predict_on_train.py --split train
else
    echo ""
    echo "[2/5] Skipping prediction (--skip-predict)"
fi

# Step 3: Rank label errors
echo ""
echo "[3/5] Ranking label errors (top $TOP)..."
"$RANKER" \
    --predictions predictions_train/ \
    --labels data/train/labels/ \
    --top "$TOP" \
    --format tsv

echo ""
echo "Full JSON report:"
"$RANKER" \
    --predictions predictions_train/ \
    --labels data/train/labels/ \
    --top "$TOP" > label_issues.json

echo "Saved to label_issues.json"

# Step 4: Local eval on validation set
echo ""
echo "[4/5] Running local mAP@0.5 on validation set..."
if [[ -d "predictions_val" ]]; then
    "$EVAL" --predictions predictions_val/ --labels data/val/labels/
else
    echo "No predictions_val/ found. Run: python scripts/predict_on_train.py --split val"
    echo "Then:  $EVAL --predictions predictions_val/ --labels data/val/labels/"
fi

# Step 5: Generate submission
echo ""
echo "[5/5] Generating submission..."
if python predict.py 2>/dev/null; then
    echo "submission.csv ready for upload"
else
    echo "predict.py not found or failed — generate submission manually"
fi

echo ""
echo "================================================"
echo "  ITERATION COMPLETE"
echo "  Review label_issues.json and fix labels"
echo "  Then run again: ./scripts/iterate.sh"
echo "================================================"
