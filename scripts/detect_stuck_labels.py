#!/usr/bin/env python3
"""
Detect "stuck labels" across all scenes in the training set.

A stuck label is a bounding box (class + coordinates) that appears with IDENTICAL
values in more than 50% of a scene's frames. These are likely copy-paste artifacts
from annotation tools where labels from one frame were propagated unchanged to
many subsequent frames -- even after the scene content changed.

For each stuck label found, we also check whether the model's predictions for
those frames agree (have a nearby detection), which helps validate whether the
label is actually wrong.
"""

import os
import re
import math
from collections import defaultdict
from pathlib import Path

LABELS_DIR = Path("/Users/william/3LC/repo/data/train/labels")
PREDS_DIR = Path("/Users/william/3LC/repo/predictions_train")
STUCK_THRESHOLD = 0.50  # Flag if label appears in > 50% of scene frames


def extract_scene_id(filename):
    """Extract MVI_XXXXX from filename like MVI_40863_img00007_jpg.rf.xxxx.txt"""
    match = re.match(r"(MVI_\d+)", filename)
    return match.group(1) if match else None


def extract_frame_number(filename):
    """Extract frame number from filename like MVI_40863_img00007_jpg.rf.xxxx.txt"""
    match = re.search(r"_img(\d+)_", filename)
    return int(match.group(1)) if match else 0


def parse_label_line(line):
    """Parse a YOLO label line: class_id x_center y_center width height"""
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    class_id = int(parts[0])
    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    return (class_id, x, y, w, h)


def parse_pred_line(line):
    """Parse a prediction line: class_id conf x_center y_center width height"""
    parts = line.strip().split()
    if len(parts) != 6:
        return None
    class_id = int(parts[0])
    conf = float(parts[1])
    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
    return (class_id, conf, x, y, w, h)


def iou(box1, box2):
    """Compute IoU between two boxes in (x_center, y_center, w, h) format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner format
    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
        return 0.0

    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def check_prediction_agreement(stuck_label, scene_files, preds_dir, iou_threshold=0.3):
    """
    For a stuck label, check how many frames have a matching prediction.
    Returns (frames_with_pred_match, frames_without_match, total_frames_checked,
             avg_match_conf, avg_match_iou)
    """
    class_id, x, y, w, h = stuck_label
    label_box = (x, y, w, h)

    frames_with_match = 0
    frames_without_match = 0
    total_checked = 0
    match_confs = []
    match_ious = []

    for label_file in scene_files:
        pred_file = preds_dir / label_file.name
        if not pred_file.exists():
            continue

        total_checked += 1
        best_iou = 0.0
        best_conf = 0.0

        with open(pred_file) as f:
            for line in f:
                pred = parse_pred_line(line)
                if pred is None:
                    continue
                pred_class, pred_conf, px, py, pw, ph = pred
                # Check same class
                if pred_class != class_id:
                    continue
                pred_box = (px, py, pw, ph)
                cur_iou = iou(label_box, pred_box)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_conf = pred_conf

        if best_iou >= iou_threshold:
            frames_with_match += 1
            match_confs.append(best_conf)
            match_ious.append(best_iou)
        else:
            frames_without_match += 1

    avg_conf = sum(match_confs) / len(match_confs) if match_confs else 0.0
    avg_iou = sum(match_ious) / len(match_ious) if match_ious else 0.0

    return frames_with_match, frames_without_match, total_checked, avg_conf, avg_iou


CLASS_NAMES = {0: "pedestrian", 1: "car", 2: "van", 3: "bus", 4: "truck"}


def main():
    print("=" * 100)
    print("STUCK LABEL DETECTOR - Scanning all scenes for copy-paste annotation artifacts")
    print("=" * 100)

    # Group files by scene
    scenes = defaultdict(list)
    all_files = sorted(LABELS_DIR.glob("*.txt"))

    for f in all_files:
        scene = extract_scene_id(f.name)
        if scene:
            scenes[scene].append(f)

    print(f"\nTotal label files: {len(all_files)}")
    print(f"Total scenes: {len(scenes)}")
    print()

    # Sort scenes by ID for consistent output
    sorted_scenes = sorted(scenes.keys())

    # Track global stats
    total_stuck_labels = 0
    total_affected_frames = 0
    total_frames_all = 0
    scenes_with_stuck = []
    all_stuck_details = []

    for scene_id in sorted_scenes:
        scene_files = sorted(scenes[scene_id], key=lambda f: extract_frame_number(f.name))
        num_frames = len(scene_files)
        total_frames_all += num_frames

        # Count occurrences of each exact label line
        label_counts = defaultdict(list)  # label_tuple -> list of filenames

        for label_file in scene_files:
            with open(label_file) as f:
                for line in f:
                    label = parse_label_line(line)
                    if label is None:
                        continue
                    label_counts[label].append(label_file)

        # Find stuck labels (appearing in > 50% of frames)
        threshold_count = math.ceil(num_frames * STUCK_THRESHOLD)
        stuck_labels = []

        for label, files_containing in label_counts.items():
            count = len(files_containing)
            if count > threshold_count and count > 3:  # Also require >3 to avoid tiny scenes
                stuck_labels.append((label, count, files_containing))

        if stuck_labels:
            scenes_with_stuck.append(scene_id)
            print(f"\n{'='*100}")
            print(f"SCENE: {scene_id} ({num_frames} frames) -- {len(stuck_labels)} STUCK LABEL(S) FOUND")
            print(f"{'='*100}")

            for label, count, files_containing in sorted(stuck_labels, key=lambda x: -x[1]):
                class_id, x, y, w, h = label
                pct = count / num_frames * 100
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                # Check prediction agreement
                match_count, nomatch_count, checked, avg_conf, avg_iou = \
                    check_prediction_agreement(label, files_containing, PREDS_DIR)

                pred_agree_pct = match_count / checked * 100 if checked > 0 else 0

                # Determine verdict
                if pred_agree_pct > 70:
                    verdict = "LIKELY VALID (predictions agree)"
                elif pred_agree_pct < 30:
                    verdict = "LIKELY STUCK/PHANTOM (predictions disagree)"
                else:
                    verdict = "UNCERTAIN (mixed prediction agreement)"

                total_stuck_labels += 1
                total_affected_frames += count

                all_stuck_details.append({
                    "scene": scene_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "x": x, "y": y, "w": w, "h": h,
                    "frames_with_label": count,
                    "total_frames": num_frames,
                    "pct": pct,
                    "pred_agree_pct": pred_agree_pct,
                    "avg_pred_conf": avg_conf,
                    "avg_pred_iou": avg_iou,
                    "verdict": verdict,
                })

                print(f"\n  Label: class={class_id} ({class_name})"
                      f"  x={x:.8f} y={y:.8f} w={w:.8f} h={h:.8f}")
                print(f"  Appears in: {count}/{num_frames} frames ({pct:.1f}%)")
                print(f"  Prediction agreement: {match_count}/{checked} frames ({pred_agree_pct:.1f}%)"
                      f"  avg_conf={avg_conf:.3f}  avg_iou={avg_iou:.3f}")
                print(f"  --> {verdict}")

                # Show first/last frame numbers to understand spread
                frame_nums = sorted([extract_frame_number(f.name) for f in files_containing])
                print(f"  Frame range: {frame_nums[0]} - {frame_nums[-1]}"
                      f"  (first 5: {frame_nums[:5]}, last 5: {frame_nums[-5:]})")

    # ==================== SUMMARY ====================
    print(f"\n\n{'='*100}")
    print("GLOBAL SUMMARY")
    print(f"{'='*100}")
    print(f"Scenes scanned:          {len(scenes)}")
    print(f"Total label files:       {len(all_files)}")
    print(f"Scenes with stuck labels: {len(scenes_with_stuck)}")
    print(f"Total stuck label types: {total_stuck_labels}")
    print(f"Total affected frame-labels: {total_affected_frames}")
    print()

    # Categorize by verdict
    likely_phantom = [d for d in all_stuck_details if "PHANTOM" in d["verdict"]]
    likely_valid = [d for d in all_stuck_details if "VALID" in d["verdict"]]
    uncertain = [d for d in all_stuck_details if "UNCERTAIN" in d["verdict"]]

    print(f"Likely PHANTOM (predictions disagree, <30% match): {len(likely_phantom)}")
    print(f"Likely VALID (predictions agree, >70% match):      {len(likely_valid)}")
    print(f"Uncertain (mixed):                                 {len(uncertain)}")

    if likely_phantom:
        print(f"\n{'='*100}")
        print("PHANTOM LABELS -- High-confidence fixes (model does NOT see objects here)")
        print(f"{'='*100}")
        phantom_frame_count = 0
        for d in sorted(likely_phantom, key=lambda x: (-x["frames_with_label"])):
            phantom_frame_count += d["frames_with_label"]
            print(f"  {d['scene']:12s}  class={d['class_id']} ({d['class_name']:10s})"
                  f"  x={d['x']:.6f} y={d['y']:.6f} w={d['w']:.6f} h={d['h']:.6f}"
                  f"  frames={d['frames_with_label']:3d}/{d['total_frames']:3d} ({d['pct']:.0f}%)"
                  f"  pred_agree={d['pred_agree_pct']:.0f}%")
        print(f"\n  TOTAL phantom frame-labels to fix: {phantom_frame_count}")

    if likely_valid:
        print(f"\n{'='*100}")
        print("LIKELY VALID -- Static objects (model predictions agree, probably real)")
        print(f"{'='*100}")
        for d in sorted(likely_valid, key=lambda x: (-x["frames_with_label"])):
            print(f"  {d['scene']:12s}  class={d['class_id']} ({d['class_name']:10s})"
                  f"  x={d['x']:.6f} y={d['y']:.6f} w={d['w']:.6f} h={d['h']:.6f}"
                  f"  frames={d['frames_with_label']:3d}/{d['total_frames']:3d} ({d['pct']:.0f}%)"
                  f"  pred_agree={d['pred_agree_pct']:.0f}%  avg_conf={d['avg_pred_conf']:.3f}")

    if uncertain:
        print(f"\n{'='*100}")
        print("UNCERTAIN -- Need manual review (mixed prediction agreement)")
        print(f"{'='*100}")
        for d in sorted(uncertain, key=lambda x: (-x["frames_with_label"])):
            print(f"  {d['scene']:12s}  class={d['class_id']} ({d['class_name']:10s})"
                  f"  x={d['x']:.6f} y={d['y']:.6f} w={d['w']:.6f} h={d['h']:.6f}"
                  f"  frames={d['frames_with_label']:3d}/{d['total_frames']:3d} ({d['pct']:.0f}%)"
                  f"  pred_agree={d['pred_agree_pct']:.0f}%  avg_conf={d['avg_pred_conf']:.3f}")

    # Per-scene breakdown
    print(f"\n{'='*100}")
    print("PER-SCENE BREAKDOWN (scenes with stuck labels only)")
    print(f"{'='*100}")
    for scene_id in scenes_with_stuck:
        scene_details = [d for d in all_stuck_details if d["scene"] == scene_id]
        phantoms = [d for d in scene_details if "PHANTOM" in d["verdict"]]
        valids = [d for d in scene_details if "VALID" in d["verdict"]]
        mixed = [d for d in scene_details if "UNCERTAIN" in d["verdict"]]
        total_scene_frames = scene_details[0]["total_frames"]
        print(f"  {scene_id}: {len(scene_details)} stuck labels"
              f" ({len(phantoms)} phantom, {len(valids)} valid, {len(mixed)} uncertain)"
              f" across {total_scene_frames} frames")


if __name__ == "__main__":
    main()
