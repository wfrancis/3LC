"""Fast training loop — bypasses 3LC tables, uses raw YOLO data directly.

This is for rapid iteration. Use train.py (with 3LC) for final submission
and Dashboard inspection.

Usage:
    python scripts/train_fast.py [--epochs 5] [--run-name fast_v1]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

WORK_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Fast YOLOv8n training (no 3LC overhead)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--run-name", default="fast_iter", help="Run name")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (auto-detect)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--resume", default=None, help="Resume from weights path")
    args = parser.parse_args()

    # Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = 0
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dataset_yaml = WORK_DIR / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"ERROR: {dataset_yaml} not found")
        sys.exit(1)

    print("=" * 60)
    print(f"FAST TRAIN: {args.epochs} epochs, device={device}, batch={args.batch}")
    print("=" * 60)

    if args.resume:
        model = YOLO(args.resume)
        print(f"Resuming from: {args.resume}")
    else:
        model = YOLO("yolov8n.yaml")
        print("Training from scratch (yolov8n.yaml)")

    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        name=args.run_name,
        project=str(WORK_DIR / "runs" / "detect"),
        exist_ok=True,
        # SPEED: cache images in RAM — eliminates disk I/O every epoch
        cache="ram",
        # SPEED: validate every 5 epochs (save_period=-1 means no extra saves)
        val=True,
        # From scratch — needs proper LR schedule
        pretrained=False,
        lr0=0.02,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=5.0,
        warmup_momentum=0.5,
        # Augmentation — stronger for small objects
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,
        scale=0.9,
        degrees=5.0,
        close_mosaic=15,
        # Regularization
        weight_decay=0.0005,
        patience=30,
        # SPEED: rectangular validation (no wasted padding)
        rect=True,
    )

    weights = WORK_DIR / "runs" / "detect" / args.run_name / "weights" / "best.pt"
    print(f"\nDone. Weights: {weights}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
