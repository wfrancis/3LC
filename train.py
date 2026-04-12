#!/usr/bin/env python3
"""
Train YOLOv8n on 3LC train/val tables. All settings come from config.yaml.

Competition rules enforced here:
- YOLOv8n only (config training.model).
- Train from scratch only (yolov8n.yaml); COCO or other pretrained checkpoints are not used.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import tlc
from tlc_ultralytics import YOLO, Settings


def _apply_ultralytics_83_compat() -> None:
    """
    Compatibility shim for Ultralytics 8.3+ with 3LC Ultralytics integration.

    In some Ultralytics versions, detection predictions can arrive as a dict
    (with keys like bboxes/conf/cls) instead of a single Nx6 tensor. The 3LC
    validator expects a tensor. Patch the validator to normalize dict -> tensor.
    """

    from tlc_ultralytics.detect import validator as det_val

    if getattr(det_val.TLCDetectionValidator, "_ua_detrac_compat_patched", False):
        return

    original = det_val.TLCDetectionValidator._process_detection_predictions

    def _pred_dict_to_tensor(predictions):
        if isinstance(predictions, torch.Tensor):
            return predictions
        bboxes = predictions["bboxes"]
        conf = predictions["conf"]
        cls = predictions["cls"]
        if bboxes.shape[0] == 0:
            return bboxes.new_empty((0, 6))
        return torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1).float()], dim=1)

    def patched(self, preds, batch):
        from tlc_ultralytics.detect.utils import construct_bbox_struct
        from ultralytics.utils import metrics, ops

        predicted_boxes = []
        for i, predictions in enumerate(preds):
            predictions = _pred_dict_to_tensor(predictions)
            ori_shape = batch["ori_shape"][i]
            resized_shape = batch["resized_shape"][i]
            ratio_pad = batch["ratio_pad"][i]
            height, width = ori_shape

            if len(predictions) == 0:
                predicted_boxes.append(
                    construct_bbox_struct(
                        [],
                        image_width=width,
                        image_height=height,
                    )
                )
                continue

            predictions = predictions.clone()
            predictions = predictions[predictions[:, 4] > self._settings.conf_thres]
            predictions = predictions[predictions[:, 4].argsort(descending=True)[: self._settings.max_det]]

            pred_box = predictions[:, :4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)

            pbatch = self._prepare_batch(i, batch)
            gt_bbox = pbatch.get("bbox", pbatch.get("bboxes"))
            if gt_bbox is not None and gt_bbox.shape[0]:
                ious = metrics.box_iou(gt_bbox, pred_scaled)
                box_ious = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0]

            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)

            conf = predictions[:, 4].cpu().tolist()
            pred_cls = predictions[:, 5].cpu().tolist()

            annotations = [
                {
                    "score": conf[pi],
                    "category_id": self.data["range_to_3lc_class"][int(pred_cls[pi])],
                    "bbox": pred_xywh[pi, :].cpu().tolist(),
                    "iou": box_ious[pi],
                }
                for pi in range(len(predictions))
            ]

            predicted_boxes.append(
                construct_bbox_struct(
                    annotations,
                    image_width=width,
                    image_height=height,
                )
            )

        return predicted_boxes

    det_val.TLCDetectionValidator._process_detection_predictions = patched
    det_val.TLCDetectionValidator._ua_detrac_compat_patched = True
    det_val.TLCDetectionValidator._ua_detrac_compat_original = original

WORK_DIR = Path(__file__).resolve().parent
_LOCKED_ARCH = "yolov8n"
_YOLOV8N_YAML = "yolov8n.yaml"


def _load_config() -> dict:
    import yaml

    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_model_stem(name: str) -> str:
    s = str(name).lower().strip()
    for suf in (".pt", ".yaml", ".yml"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def _assert_yolov8n_only(training: dict) -> None:
    raw = training.get("model")
    if raw is None:
        return
    stem = _normalize_model_stem(str(raw))
    if stem != _LOCKED_ARCH:
        raise SystemExit(
            f"train.py is locked to YOLOv8n only. config training.model={raw!r} "
            f"resolves to {stem!r}. Remove training.model or set it to yolov8n."
        )


def _apply_seed(seed: int | None) -> None:
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _reject_pretrained_config(training: dict) -> None:
    if "pretrained" not in training:
        return
    if training["pretrained"]:
        raise SystemExit(
            "Competition starter does not allow pretrained weights. Remove "
            "`training.pretrained` from config.yaml (training is always from scratch "
            f"via {_YOLOV8N_YAML})."
        )


def _check_umap(emb_dim: int, reducer: str) -> None:
    """Fail fast if umap-learn is needed but missing (avoids crash after training)."""
    if emb_dim <= 0 or reducer != "umap":
        return
    try:
        import umap  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\n  umap-learn is required for image embeddings but is not installed.\n"
            "  Fix:  pip install umap-learn\n"
            "  Or:   set image_embeddings_dim: 0 in config.yaml to skip embeddings.\n"
        )


def main() -> int:
    os.chdir(WORK_DIR)
    _apply_ultralytics_83_compat()
    cfg = _load_config()
    tlc_cfg = cfg.get("tlc", {})
    training = cfg.get("training", {})
    repro = cfg.get("reproducibility", {})

    _assert_yolov8n_only(training)
    _reject_pretrained_config(training)
    _apply_seed(repro.get("seed"))

    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    val_name = str(tlc_cfg.get("val_table_name", f"{dataset_name}-val"))
    emb_dim = int(tlc_cfg.get("image_embeddings_dim", 3))
    emb_reducer = str(tlc_cfg.get("image_embeddings_reducer", "umap"))
    _check_umap(emb_dim, emb_reducer)

    run_name = str(training.get("run_name", "yolov8n_baseline"))
    run_desc = str(training.get("run_description", ""))
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 16))
    image_size = int(training.get("image_size", 640))
    device = training.get("device", 0)
    workers = int(training.get("workers", 4))
    lr0 = float(training.get("lr0", 0.01))
    use_aug = bool(training.get("use_augmentation", True))

    print("=" * 70)
    print("TRAINING (YOLOv8n + 3LC)")
    print("=" * 70)
    print(f"\n  PyTorch: {torch.__version__}, 3LC: {tlc.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")

    print("\n  Loading tables (.latest() — includes Dashboard edits)...")
    train_table = tlc.Table.from_names(
        project_name=project_name, dataset_name=dataset_name, table_name=train_name
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=project_name, dataset_name=dataset_name, table_name=val_name
    ).latest()
    print(f"  Train: {len(train_table)} | Val: {len(val_table)}")

    tables_used = WORK_DIR / "tables_used.txt"
    tables_used.write_text(
        f"train_url={train_table.url}\nval_url={val_table.url}\n",
        encoding="utf-8",
    )
    print(f"  Wrote {tables_used}")

    model_path = _YOLOV8N_YAML
    print(
        f"\n  Model: YOLOv8n from scratch ({model_path}) | run: {run_name} | "
        f"epochs: {epochs} | batch: {batch_size} | imgsz: {image_size} | aug: {use_aug}"
    )

    settings = Settings(
        project_name=project_name,
        run_name=run_name,
        run_description=run_desc,
        image_embeddings_dim=emb_dim,
        image_embeddings_reducer=emb_reducer,
    )

    model = YOLO(model_path)
    train_args: dict = {
        "tables": {"train": train_table, "val": val_table},
        "name": run_name,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "lr0": lr0,
        "settings": settings,
        "val": True,
    }
    if use_aug:
        train_args.update({"mosaic": 1.0, "mixup": 0.05, "copy_paste": 0.1})

    model.train(**train_args)

    runs_root = cfg.get("paths", {}).get("runs_detect_root", "runs/detect")
    print("\n" + "=" * 70)
    print("OK — TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Weights: {WORK_DIR / runs_root / run_name / 'weights' / 'best.pt'}")
    print("  Next: python predict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
