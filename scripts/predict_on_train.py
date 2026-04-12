"""Run YOLOv8n inference on the training set and save predictions as YOLO-format txt files.

Usage:
    python scripts/predict_on_train.py [--config config.yaml] [--weights runs/detect/.../weights/best.pt]

Output: predictions_train/ directory with one .txt file per image, format:
    class_id confidence x_center y_center width height
(YOLO normalized coordinates)
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Predict on training images for label analysis")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", default=None, help="Path to best.pt (auto-detected from latest run if omitted)")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Which split to predict on")
    parser.add_argument("--output", default=None, help="Output directory for prediction txts")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (auto-detected if omitted)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Find images directory
    data_root = config.get("data_root", "data")
    img_dir = Path(data_root) / args.split / "images"
    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        sys.exit(1)

    # Find weights
    weights = args.weights
    if weights is None:
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            runs = sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            for run in runs:
                w = run / "weights" / "best.pt"
                if w.exists():
                    weights = str(w)
                    break
    if weights is None or not Path(weights).exists():
        print("ERROR: No weights found. Train first or specify --weights")
        sys.exit(1)

    print(f"Weights: {weights}")
    print(f"Images:  {img_dir}")

    # Output directory
    out_dir = Path(args.output) if args.output else Path(f"predictions_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = YOLO(weights)

    # Get all images
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in extensions])
    print(f"Found {len(images)} images")

    # Run inference in batches
    count = 0
    for i in range(0, len(images), args.batch):
        batch_paths = images[i : i + args.batch]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=args.conf,
            device=device,
            verbose=False,
        )

        for path, result in zip(batch_paths, results):
            stem = path.stem
            out_file = out_dir / f"{stem}.txt"

            lines = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                xywhn = boxes.xywhn.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                for cls, conf, box in zip(classes, confs, xywhn):
                    x, y, w, h = box
                    lines.append(f"{cls} {conf:.6f} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

            out_file.write_text("\n".join(lines) + ("\n" if lines else ""))
            count += 1

        done = min(i + args.batch, len(images))
        print(f"  {done}/{len(images)} images processed", end="\r")

    print(f"\nDone. Wrote {count} prediction files to {out_dir}/")


if __name__ == "__main__":
    main()
