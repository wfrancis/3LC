#!/usr/bin/env python3
"""
Run inference on test images and write Kaggle submission CSV.

Same inference pattern as the main repo predict.py (memory = chunked in-GPU predict;
txt = save_txt + label files). Paths and hyperparameters: config.yaml only.

Default pipeline is **memory** — best for ~8GB GPUs after training (avoids save_txt stream spikes).
Set predict.pipeline: txt in config if you prefer the cotton-style label export.
"""

from __future__ import annotations

import csv
import gc
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from tlc_ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


WORK_DIR = Path(__file__).resolve().parent


def _apply_ultralytics_83_compat() -> None:
    """
    Compatibility shim for Ultralytics 8.3+ with 3LC Ultralytics integration.

    See train.py for details. This must run before creating a YOLO() model so
    that validation/collection paths can handle dict-style predictions.
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


def _load_config() -> dict:
    import yaml

    p = WORK_DIR / "config.yaml"
    if not p.is_file():
        print(f"ERROR: Missing {p}", file=sys.stderr)
        sys.exit(1)
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (WORK_DIR / p).resolve()


def _resolve_weights(cfg: dict) -> Path:
    paths_cfg = cfg.get("paths", {})
    training = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})
    runs_root = str(paths_cfg.get("runs_detect_root", "runs/detect"))
    run_name = str(training.get("run_name", "yolov8n_baseline"))

    explicit = predict_cfg.get("weights")
    if explicit:
        p = _resolve(str(explicit))
        if p.is_file():
            return p

    primary = WORK_DIR / runs_root / run_name / "weights" / "best.pt"
    if primary.is_file():
        return primary.resolve()

    candidates = sorted(
        (WORK_DIR / runs_root).glob("*/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "No weights found. Run train.py first or set predict.weights in config.yaml."
    )


def _label_file_to_prediction_string(label_path: Path) -> str:
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return "no box"
    tokens: list[str] = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id, xc, yc, w, h, conf = (
                    parts[0],
                    parts[1],
                    parts[2],
                    parts[3],
                    parts[4],
                    parts[5],
                )
                tokens.append(f"{class_id} {conf} {xc} {yc} {w} {h}")
    return " ".join(tokens) if tokens else "no box"


def _result_to_prediction_string(result) -> str:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no box"
    xywhn = boxes.xywhn.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    order = np.argsort(-conf)
    parts: list[str] = []
    for i in order:
        c = int(cls[i])
        cf = float(conf[i])
        x, y, w, h = (float(v) for v in xywhn[i])
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        w = min(1.0, max(0.0, w))
        h = min(1.0, max(0.0, h))
        parts.extend(
            [str(c), f"{cf:.6f}", f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"]
        )
    return " ".join(parts)


def _find_image(test_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = test_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def _pipeline_txt(
    model: YOLO,
    test_dir: Path,
    stems: list[str],
    pred_dir_name: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    pred_root = WORK_DIR / pred_dir_name
    if pred_root.exists():
        shutil.rmtree(pred_root)

    paths: list[Path] = []
    for stem in stems:
        p = _find_image(test_dir, stem)
        if p is not None:
            paths.append(p)

    if not paths:
        raise RuntimeError(
            f"No test images found for sample submission image_ids under {test_dir}"
        )

    bs = max(1, int(batch))
    results = model.predict(
        source=[str(p) for p in paths],
        save=False,
        save_txt=True,
        save_conf=True,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        batch=bs,
        project=str(WORK_DIR),
        name=pred_dir_name,
        exist_ok=False,
        verbose=False,
        stream=True,
    )
    for _ in tqdm(results, total=len(paths), desc="Predicting", unit="img"):
        pass

    labels_dir = pred_root / "labels"
    pred_by_stem: dict[str, str] = {}
    for stem in stems:
        pred_by_stem[stem] = _label_file_to_prediction_string(labels_dir / f"{stem}.txt")

    if pred_root.exists():
        shutil.rmtree(pred_root)

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return pred_by_stem


def _pipeline_memory(
    model: YOLO,
    test_dir: Path,
    stems: list[str],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    paths: list[Path] = []
    for stem in stems:
        p = _find_image(test_dir, stem)
        if p is not None:
            paths.append(p)

    pred_by_stem: dict[str, str] = {}
    bs = max(1, int(batch))
    n = len(paths)
    n_chunks = (n + bs - 1) // bs
    for start in tqdm(
        range(0, n, bs),
        total=n_chunks,
        desc="Predicting",
        unit="batch",
    ):
        chunk = paths[start : start + bs]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            device=device,
            max_det=int(max_det),
            batch=min(bs, len(chunk)),
            verbose=False,
        )
        for res, p in zip(results, chunk):
            pred_by_stem[p.stem] = _result_to_prediction_string(res)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    return pred_by_stem


def main() -> int:
    os.chdir(WORK_DIR)
    _apply_ultralytics_83_compat()
    cfg = _load_config()
    paths_cfg = cfg.get("paths", {})
    training = cfg.get("training", {})
    predict_cfg = cfg.get("predict", {})

    sample_path = _resolve(str(paths_cfg.get("sample_submission", "sample_submission.csv")))
    out_path = _resolve(str(paths_cfg.get("submission_csv", "submission.csv")))
    test_dir = _resolve(str(paths_cfg.get("test_images", "data/test/images")))

    conf = float(predict_cfg.get("conf", 0.25))
    iou = float(predict_cfg.get("iou", 0.7))
    imgsz = predict_cfg.get("imgsz")
    if imgsz is None:
        imgsz = int(training.get("image_size", 640))
    else:
        imgsz = int(imgsz)
    device = (
        predict_cfg["device"]
        if predict_cfg.get("device") is not None
        else training.get("device", 0)
    )
    max_det = int(predict_cfg.get("max_det", 300))
    batch = int(predict_cfg.get("batch", 1))
    pipeline = str(predict_cfg.get("pipeline", "memory")).lower()
    pred_dir_name = str(predict_cfg.get("pred_dir_name", "predictions"))

    print("=" * 70)
    print("PREDICTIONS → SUBMISSION CSV")
    print("=" * 70)

    if not sample_path.is_file():
        print(f"ERROR: sample submission not found: {sample_path}", file=sys.stderr)
        return 1
    if not test_dir.is_dir():
        print(f"ERROR: test images directory not found: {test_dir}", file=sys.stderr)
        return 1

    try:
        weights = _resolve_weights(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    rows_in: list[dict] = []
    with sample_path.open(newline="", encoding="utf-8") as f:
        rows_in.extend(csv.DictReader(f))

    stems = [str(r["image_id"]).strip() for r in rows_in]
    print(f"\n  Rows (sample): {len(rows_in)}")
    print(f"  Test dir: {test_dir}")
    print(f"  Weights: {weights}")
    print(f"  Pipeline: {pipeline}")
    print(f"  conf={conf}, iou={iou}, imgsz={imgsz}, batch={batch}")

    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    model = YOLO(str(weights))
    print("  Model loaded.")

    if pipeline == "txt":
        pred_by_stem = _pipeline_txt(
            model,
            test_dir,
            stems,
            pred_dir_name,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            batch=batch,
        )
    else:
        pred_by_stem = _pipeline_memory(
            model,
            test_dir,
            stems,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            max_det=max_det,
            batch=batch,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "image_id", "prediction_string"])
        w.writeheader()
        for row, stem in tqdm(
            zip(rows_in, stems),
            total=len(rows_in),
            desc="Writing CSV",
            unit="row",
        ):
            w.writerow(
                {
                    "id": row["id"],
                    "image_id": stem,
                    "prediction_string": pred_by_stem.get(stem, "no box"),
                }
            )

    nonempty = sum(
        1 for stem in stems if pred_by_stem.get(stem, "no box") != "no box"
    )
    n_boxes = sum(
        len(pred_by_stem.get(stem, "").split()) // 6
        for stem in stems
        if pred_by_stem.get(stem, "no box") != "no box"
    )

    print("\n" + "=" * 70)
    print("OK — SUBMISSION READY")
    print("=" * 70)
    print(f"  File: {out_path}")
    print(f"  Images with ≥1 box: {nonempty} / {len(rows_in)}")
    print(f"  Total boxes (approx): {n_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
