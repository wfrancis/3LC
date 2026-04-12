#!/usr/bin/env bash
# Quick submit: predict on test set → submission.csv → Kaggle upload
# Usage: ./scripts/submit.sh [--weights path/to/best.pt] [--message "description"]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source 3lc-env/bin/activate

WEIGHTS=""
MESSAGE="auto submission"

while [[ $# -gt 0 ]]; do
    case $1 in
        --weights) WEIGHTS="$2"; shift 2 ;;
        --message) MESSAGE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

# Auto-find weights if not specified
if [[ -z "$WEIGHTS" ]]; then
    # Check fast_iter first, then yolov8n_baseline, then newest
    for candidate in \
        runs/detect/fast_iter/weights/best.pt \
        /Users/william/SportAI/runs/detect/yolov8n_baseline/weights/best.pt \
        runs/detect/yolov8n_baseline/weights/best.pt; do
        if [[ -f "$candidate" ]]; then
            WEIGHTS="$candidate"
            break
        fi
    done
fi

if [[ -z "$WEIGHTS" ]] || [[ ! -f "$WEIGHTS" ]]; then
    echo "ERROR: No weights found. Specify --weights or train first."
    exit 1
fi

echo "=== SUBMIT ==="
echo "Weights: $WEIGHTS"
echo "Message: $MESSAGE"

# Temporarily update config to point to these weights
# predict.py reads from config, but we can also just use it directly
python predict.py

if [[ ! -f submission.csv ]]; then
    echo "ERROR: submission.csv not created"
    exit 1
fi

echo ""
echo "submission.csv ready. Lines: $(wc -l < submission.csv)"
echo ""
echo "To upload to Kaggle:"
echo "  kaggle competitions submit -c 3-lc-multi-vehicle-detection-challenge -f submission.csv -m \"$MESSAGE\""

# Try auto-upload if kaggle CLI is available
if command -v kaggle &>/dev/null; then
    echo ""
    read -p "Upload now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kaggle competitions submit \
            -c 3-lc-multi-vehicle-detection-challenge \
            -f submission.csv \
            -m "$MESSAGE"
        echo "Submitted!"
    fi
else
    echo ""
    echo "Install kaggle CLI: pip install kaggle"
fi
