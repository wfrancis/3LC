---
description: "One-command full iteration: rank label errors, review with vision, fix, and optionally retrain"
---

# /fix-labels

Full iteration cycle for the 3LC competition. Run this to find and fix label errors, then retrain.

## Steps

1. **Check for trained model** -- look for `runs/detect/*/weights/best.pt`
2. **Predict on training set** -- run `python scripts/predict_on_train.py` to generate prediction files
3. **Run label-ranker** -- `./rust/target/release/label-ranker --predictions predictions_train/ --labels data/train/labels/ --top 50`
4. **Review flagged images** -- spawn the label-fixer agent to review the top flagged images using vision
5. **Report changes** -- summarize what was fixed
6. **Optionally retrain** -- ask the user if they want to retrain now
7. **Optionally predict + eval** -- run predict.py and map-eval to check local score

## Usage

Just type `/fix-labels` to start the full cycle.

If you want to skip to a specific step:
- `/fix-labels rank` -- just run the ranker
- `/fix-labels review` -- just review previously flagged images
- `/fix-labels eval` -- just run local evaluation

## Paths (from config.yaml)

- Training images: `data/train/images/`
- Training labels: `data/train/labels/`
- Predictions output: `predictions_train/`
- Rust binaries: `rust/target/release/`
