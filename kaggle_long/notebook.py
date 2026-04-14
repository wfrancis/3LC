#!/usr/bin/env python3
"""
Kaggle Training Notebook — 3LC Multi-Vehicle Detection Challenge
=================================================================

Designed for Kaggle's free P100 GPU (16 GB VRAM).

This script:
  1. Installs required packages (3lc-ultralytics, umap-learn)
  2. Clones the competition repo
  3. Copies Kaggle competition data into the expected directory layout
  4. Applies 3,623 label cleaning fixes through cycle 3 (tiny boxes, phantom/stuck
     labels, wrong classes, duplicate labels, inaccurate bounding boxes,
     scene-specific fixes) — the core advantage of our pipeline.
     Cycle 5 fixes are intentionally omitted here (they hurt performance).
  5. Trains YOLOv8n from scratch with optimized hyperparameters (200 epochs,
     patience=75) for a long run
  6. Generates predictions on the test set and writes submission.csv
  7. Saves best weights as a Kaggle output artifact

Usage:
  Paste the contents of each section into a Kaggle notebook cell,
  or run the entire file as a single cell.

Competition rules enforced:
  - YOLOv8n architecture only, trained from scratch (yolov8n.yaml)
  - 640px input, no pretrained weights, no ensembles/TTA/pseudo-labels
"""

# ============================================================================
# SECTION 0: Install packages
# ============================================================================

import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# Kaggle's PyTorch 2.10+cu128 drops P100 (sm_60). Reinstall with CUDA 11.8.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "torch==2.5.1", "torchvision==0.20.1",
    "--index-url", "https://download.pytorch.org/whl/cu118"
])
install("ultralytics==8.3.40")


print("Packages installed.")

# Check what ultralytics version Kaggle has
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Disable ray to avoid conflict with ultralytics on Kaggle
import sys as _sys
_sys.modules["ray"] = type(_sys)("ray")
_sys.modules["ray.tune"] = type(_sys)("ray.tune")
_sys.modules["ray.train"] = type(_sys)("ray.train")


# ============================================================================
# SECTION 1: Clone repo and set up directories
# ============================================================================

import os
import shutil
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input/3-lc-multi-vehicle-detection-challenge")
WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "3LC"
DATA_DIR = WORK_DIR / "data"

# Clone our repo (contains cleaning scripts, config, submission tooling)
os.chdir(str(WORK_DIR))
if not REPO_DIR.exists():
    subprocess.check_call(
        ["git", "clone", "https://github.com/wfrancis/3LC.git"],
        cwd=str(WORK_DIR),
    )
    print(f"Repo cloned to {REPO_DIR}")
else:
    print(f"Repo already exists at {REPO_DIR}")


# ============================================================================
# SECTION 2: Copy competition data from Kaggle input to working directory
# ============================================================================

# Kaggle mounts competition data read-only under /kaggle/input/.
# We need a writable copy so we can apply label fixes.
#
# Expected Kaggle input structure:
#   /kaggle/input/3-lc-multi-vehicle-detection-challenge/
#     train/images/
#     train/labels/
#     val/images/
#     val/labels/
#     test/images/
#     sample_submission.csv

# Auto-detect Kaggle input structure (may be nested differently)
import glob as _glob

def find_split_dir(base, split, subdir):
    """Find train/images, val/labels etc. under base, handling nested structures."""
    # Try direct: base/train/images
    direct = base / split / subdir
    if direct.is_dir():
        return direct
    # Try nested: base/*/train/images or base/*/*/train/images
    for pattern in [f"*/{split}/{subdir}", f"*/*/{split}/{subdir}", f"**/{split}/{subdir}"]:
        matches = sorted(base.glob(pattern))
        if matches:
            return matches[0]
    return direct  # fallback, will error with clear message

# List everything in /kaggle/input to find the actual data path
kaggle_base = Path("/kaggle/input")
print(f"  Scanning /kaggle/input/:")
if kaggle_base.exists():
    for item in sorted(kaggle_base.iterdir()):
        print(f"    {item.name}/ ({len(list(item.iterdir())) if item.is_dir() else 'file'})")
        if item.is_dir():
            for sub in sorted(item.iterdir())[:15]:
                print(f"      {sub.name}/ ({len(list(sub.iterdir())) if sub.is_dir() else 'file'})")
else:
    print("    /kaggle/input/ does not exist!")

# Auto-detect the competition data root
KAGGLE_INPUT = None
for candidate in [
    kaggle_base / "3-lc-multi-vehicle-detection-challenge",
    kaggle_base / "3lc-multi-vehicle-detection-challenge",
    kaggle_base / "competitions" / "3-lc-multi-vehicle-detection-challenge",
    kaggle_base / "competitions" / "3lc-multi-vehicle-detection-challenge",
]:
    if candidate.exists():
        KAGGLE_INPUT = candidate
        break

# If not found by name, find any directory containing train/images
if KAGGLE_INPUT is None:
    for item in kaggle_base.iterdir():
        if item.is_dir():
            if (item / "train" / "images").exists():
                KAGGLE_INPUT = item
                break
            # Check one level deeper
            for sub in item.iterdir():
                if sub.is_dir() and (sub / "train" / "images").exists():
                    KAGGLE_INPUT = sub
                    break
            if KAGGLE_INPUT:
                break

if KAGGLE_INPUT is None:
    raise FileNotFoundError(
        "Could not find competition data under /kaggle/input/. "
        "Make sure the competition dataset is attached to the notebook."
    )

print(f"  Found competition data at: {KAGGLE_INPUT}")
print(f"  Contents: {[p.name for p in KAGGLE_INPUT.iterdir()]}")

# Handle competition_starter/ nesting
if not (KAGGLE_INPUT / "train").exists() and (KAGGLE_INPUT / "competition_starter").exists():
    KAGGLE_INPUT = KAGGLE_INPUT / "competition_starter"
    print(f"  Adjusted to: {KAGGLE_INPUT}")
if not (KAGGLE_INPUT / "train").exists() and (KAGGLE_INPUT / "data").exists():
    KAGGLE_INPUT = KAGGLE_INPUT / "data"
    print(f"  Adjusted to: {KAGGLE_INPUT}")

for split in ("train", "val", "test"):
    src_images = find_split_dir(KAGGLE_INPUT, split, "images")
    dst_images = DATA_DIR / split / "images"
    dst_images.mkdir(parents=True, exist_ok=True)

    if not any(dst_images.iterdir()):
        if not src_images.is_dir():
            print(f"  WARNING: {src_images} not found — skipping {split}/images")
            continue
        # Symlink images (read-only is fine, saves disk and time)
        for img in src_images.iterdir():
            dst = dst_images / img.name
            if not dst.exists():
                os.symlink(str(img), str(dst))
        print(f"  Linked {split}/images ({len(list(dst_images.iterdir()))} files)")
    else:
        print(f"  {split}/images already populated")

    # Labels need to be writable copies (we will modify them)
    if split == "test":
        continue  # test has no labels
    src_labels = find_split_dir(KAGGLE_INPUT, split, "labels")
    dst_labels = DATA_DIR / split / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)

    if not any(dst_labels.iterdir()):
        for lbl in src_labels.iterdir():
            shutil.copy2(str(lbl), str(dst_labels / lbl.name))
        print(f"  Copied {split}/labels ({len(list(dst_labels.iterdir()))} files)")
    else:
        print(f"  {split}/labels already populated")

# Copy sample_submission.csv (may be in KAGGLE_INPUT, parent, or grandparent)
sample_sub_dst = WORK_DIR / "sample_submission.csv"
if not sample_sub_dst.exists():
    for search_dir in [KAGGLE_INPUT, KAGGLE_INPUT.parent, KAGGLE_INPUT.parent.parent,
                       kaggle_base / "competitions" / "3-lc-multi-vehicle-detection-challenge"]:
        candidate = search_dir / "sample_submission.csv"
        if candidate.exists():
            shutil.copy2(str(candidate), str(sample_sub_dst))
            print(f"  Copied sample_submission.csv from {search_dir}")
            break
        # Also check inside competition_starter/
        candidate2 = search_dir / "competition_starter" / "sample_submission.csv"
        if candidate2.exists():
            shutil.copy2(str(candidate2), str(sample_sub_dst))
            print(f"  Copied sample_submission.csv from {candidate2.parent}")
            break
if not sample_sub_dst.exists():
    print("  WARNING: sample_submission.csv not found!")

print("Data setup complete.")


# ============================================================================
# SECTION 3: Apply all label cleaning fixes (3,623 total fixes)
# ============================================================================
#
# Our pipeline identified and fixed 6 categories of label errors in the
# original competition data. Since the cleaned labels are not committed to
# the repo (they are local modifications), we apply the exact same fixes
# programmatically here.
#
# Fix categories:
#   1. Tiny boxes:     Remove labels where w * h < 0.001 (< 0.1% of image area)
#                      These are sub-pixel noise the model cannot learn from.
#                      ~1,045 labels removed from train, ~515 from val.
#
#   2. Stuck/phantom labels (5 scenes): Labels copy-pasted across many frames
#                      with identical coordinates, even after the vehicle left.
#                      Identified by appearing in >50% of a scene's frames with
#                      <30% prediction agreement from a trained model.
#                      ~896 labels removed across MVI_40863, MVI_40742,
#                      MVI_40774, MVI_40775, MVI_40904.
#
#   3. Wrong class:    10 labels with incorrect class IDs (e.g., truck labeled
#                      as car, car labeled as van). Class ID corrected in-place.
#
#   4. Duplicate labels: 3 exact duplicate label lines removed.
#
#   5. Inaccurate boxes: 24 bounding boxes with imprecise coordinates adjusted
#                        to better fit the actual vehicle boundaries.
#
#   6. Scene-specific:  2 files with mixed issues (phantom removal + missing
#                       annotation added).

import re
from collections import defaultdict


def count_labels(labels_dir):
    """Count total label lines across all files."""
    total = 0
    for f in Path(labels_dir).glob("*.txt"):
        with open(f) as fh:
            total += sum(1 for line in fh if line.strip())
    return total


TRAIN_LABELS = DATA_DIR / "train" / "labels"
VAL_LABELS = DATA_DIR / "val" / "labels"

before_train = count_labels(TRAIN_LABELS)
before_val = count_labels(VAL_LABELS)

# Delete any .cache files so YOLO rebuilds the dataset index after our edits
for cache in DATA_DIR.rglob("*.cache"):
    cache.unlink()
    print(f"  Removed cache: {cache}")

# ---- Fix 1: Remove tiny boxes (area < 0.001) from train and val ----

def remove_tiny_boxes(labels_dir, area_threshold=0.001):
    """Remove label lines where normalized w * h < threshold."""
    total_removed = 0
    files_modified = 0
    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        with open(label_file) as f:
            lines = f.readlines()
        new_lines = []
        removed = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                w, h = float(parts[3]), float(parts[4])
                if w * h < area_threshold:
                    removed += 1
                    continue
            new_lines.append(line)
        if removed > 0:
            with open(label_file, "w") as f:
                f.writelines(new_lines)
            total_removed += removed
            files_modified += 1
    return total_removed, files_modified

tiny_train, tiny_train_files = remove_tiny_boxes(TRAIN_LABELS)
tiny_val, tiny_val_files = remove_tiny_boxes(VAL_LABELS)
print(f"Fix 1 - Tiny boxes: removed {tiny_train} from train ({tiny_train_files} files), "
      f"{tiny_val} from val ({tiny_val_files} files)")


# ---- Fix 2: Remove stuck/phantom labels (5 scenes) ----

def extract_scene_id(filename):
    match = re.match(r"(MVI_\d+)", filename)
    return match.group(1) if match else None

def extract_frame_number(filename):
    match = re.search(r"img(\d+)", filename)
    return int(match.group(1)) if match else 0

def remove_phantom_labels_by_coords(labels_dir, scene_id, phantom_tuples):
    """Remove specific label lines (by exact class + coordinate match) from a scene."""
    total_removed = 0
    files_modified = 0
    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        if extract_scene_id(label_file.name) != scene_id:
            continue
        with open(label_file) as f:
            lines = f.readlines()
        new_lines = []
        removed = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if (cls, x, y, w, h) in phantom_tuples:
                        removed += 1
                        continue
                except (ValueError, IndexError):
                    pass
            new_lines.append(line)
        if removed > 0:
            with open(label_file, "w") as f:
                f.writelines(new_lines)
            total_removed += removed
            files_modified += 1
    return total_removed, files_modified

def remove_phantom_labels_after_frame(labels_dir, scene_id, after_frame, stuck_line_strings):
    """Remove stuck labels from a scene, but only from frames after a transition point."""
    total_removed = 0
    files_modified = 0
    for label_file in sorted(Path(labels_dir).glob("*.txt")):
        if extract_scene_id(label_file.name) != scene_id:
            continue
        frame_num = extract_frame_number(label_file.name)
        if frame_num <= after_frame:
            continue
        with open(label_file) as f:
            lines = f.readlines()
        new_lines = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if stripped in stuck_line_strings:
                removed += 1
                continue
            new_lines.append(line)
        if removed > 0:
            with open(label_file, "w") as f:
                f.writelines(new_lines)
            total_removed += removed
            files_modified += 1
    return total_removed, files_modified


# Scene MVI_40863: stuck labels after frame 1178 (scene transition)
MVI_40863_STUCK = {
    "2 0.78671875 0.4890625 0.2 0.228125",
    "2 0.396875 0.36015625 0.11796875 0.140625",
    "1 0.55078125 0.42890625 0.1625 0.14609375",
    "1 0.27265625 0.33984375 0.10546875 0.1015625",
    "1 0.396875 0.36015625 0.11796875 0.140625",  # variant with reclassified class
}
r, f = remove_phantom_labels_after_frame(TRAIN_LABELS, "MVI_40863", 1178, MVI_40863_STUCK)
print(f"Fix 2a - MVI_40863 stuck labels: removed {r} from {f} files")

# Scene MVI_40742: phantom labels (bus + car ghosts)
MVI_40742_PHANTOMS = {
    (3, 0.07109375, 0.35468750, 0.13984375, 0.14062500),
    (3, 0.36640625, 0.54687500, 0.41250000, 0.41093750),
    (1, 0.44296875, 0.40546875, 0.09453125, 0.09296875),
    (1, 0.58828125, 0.46015625, 0.12265625, 0.11484375),
    (1, 0.66953125, 0.43125000, 0.12187500, 0.10390625),
}
r, f = remove_phantom_labels_by_coords(TRAIN_LABELS, "MVI_40742", MVI_40742_PHANTOMS)
print(f"Fix 2b - MVI_40742 phantom labels: removed {r} from {f} files")

# Scene MVI_40774: phantom labels (car + bus ghosts)
MVI_40774_PHANTOMS = {
    (1, 0.62656250, 0.37500000, 0.08828125, 0.15000000),
    (1, 0.74765625, 0.40000000, 0.12187500, 0.16328125),
    (1, 0.67265625, 0.56171875, 0.11015625, 0.19609375),
    (3, 0.89921875, 0.45625000, 0.20078125, 0.54843750),
}
r, f = remove_phantom_labels_by_coords(TRAIN_LABELS, "MVI_40774", MVI_40774_PHANTOMS)
print(f"Fix 2c - MVI_40774 phantom labels: removed {r} from {f} files")

# Scene MVI_40904: phantom labels (pedestrian + car + van ghosts)
MVI_40904_PHANTOMS = {
    (0, 0.58515625, 0.65312500, 0.27890625, 0.25000000),
    (1, 0.36875000, 0.20703125, 0.07109375, 0.05937500),
    (2, 0.32500000, 0.22343750, 0.08984375, 0.07968750),
}
r, f = remove_phantom_labels_by_coords(TRAIN_LABELS, "MVI_40904", MVI_40904_PHANTOMS)
print(f"Fix 2d - MVI_40904 phantom labels: removed {r} from {f} files")

# Scene MVI_40775: phantom label (tiny car ghost)
MVI_40775_PHANTOMS = {
    (1, 0.46406250, 0.08046875, 0.05390625, 0.07421875),
}
r, f = remove_phantom_labels_by_coords(TRAIN_LABELS, "MVI_40775", MVI_40775_PHANTOMS)
print(f"Fix 2e - MVI_40775 phantom labels: removed {r} from {f} files")


# ---- Fix 3: Correct wrong class IDs (10 labels) ----

# Each entry: (filename_substring, old_line, new_line)
# These are exact line replacements where the class ID was wrong.
WRONG_CLASS_FIXES = [
    # MVI_40714: truck (0) should be car (1)
    ("MVI_40714_img00286", "0 0.90546875 0.35859375 0.06171875 0.09609375",
                           "1 0.90546875 0.35859375 0.06171875 0.09609375"),
    # MVI_40864: car (1) should be van (2)
    ("MVI_40864_img00349", "1 0.91015625 0.44453125 0.17890625 0.16640625",
                           "2 0.91015625 0.44453125 0.17890625 0.16640625"),
    ("MVI_40864_img00837", "1 0.9015625 0.7046875 0.196875 0.23125",
                           "2 0.9015625 0.7046875 0.196875 0.23125"),
    # MVI_40891: truck (0) should be car (1)
    ("MVI_40891_img00008", "0 0.66875 0.2546875 0.07109375 0.1",
                           "1 0.66875 0.2546875 0.07109375 0.1"),
    ("MVI_40891_img00029", "0 0.66953125 0.25625 0.07265625 0.1",
                           "1 0.66953125 0.25625 0.07265625 0.1"),
    ("MVI_40891_img00045", "0 0.66953125 0.253125 0.071875 0.1015625",
                           "1 0.66953125 0.253125 0.071875 0.1015625"),
    ("MVI_40891_img00057", "0 0.66796875 0.25390625 0.07265625 0.1",
                           "1 0.66796875 0.25390625 0.07265625 0.1"),
    ("MVI_40891_img00092", "0 0.66796875 0.25390625 0.07265625 0.1",
                           "1 0.66796875 0.25390625 0.07265625 0.1"),
    # MVI_40892: truck (0) should be car (1)
    ("MVI_40892_img01524", "0 0.66953125 0.28203125 0.075 0.10546875",
                           "1 0.66953125 0.28203125 0.075 0.10546875"),
    ("MVI_40892_img01534", "0 0.66953125 0.28359375 0.075 0.10546875",
                           "1 0.66953125 0.28359375 0.075 0.10546875"),
]

def apply_line_replacements(labels_dir, fixes):
    """Replace specific lines in label files. Each fix is (filename_prefix, old_line, new_line)."""
    total_fixed = 0
    for prefix, old_line, new_line in fixes:
        for label_file in Path(labels_dir).glob("*.txt"):
            if prefix not in label_file.name:
                continue
            with open(label_file) as f:
                content = f.read()
            old_stripped = old_line.strip()
            if old_stripped in content:
                lines = content.splitlines()
                new_lines = []
                for line in lines:
                    if line.strip() == old_stripped:
                        new_lines.append(new_line)
                        total_fixed += 1
                    else:
                        new_lines.append(line)
                with open(label_file, "w") as f:
                    f.write("\n".join(new_lines) + "\n")
    return total_fixed

wrong_class_count = apply_line_replacements(TRAIN_LABELS, WRONG_CLASS_FIXES)
print(f"Fix 3 - Wrong class IDs: corrected {wrong_class_count} labels")


# ---- Fix 4: Remove duplicate label lines (3 files) ----

DUPLICATE_FIXES = [
    # (filename_prefix, exact_duplicate_line_to_remove)
    ("MVI_40762_img00330", "1 0.1921875 0.31484375 0.06875 0.06875"),
    ("MVI_40853_img00261", "1 0.78515625 0.25 0.04765625 0.05"),
    ("MVI_40903_img00887", "1 0.80078125 0.13203125 0.06328125 0.059375"),
]

def remove_duplicate_lines(labels_dir, fixes):
    """Remove one occurrence of an exact duplicate line from each specified file."""
    total_removed = 0
    for prefix, dup_line in fixes:
        for label_file in Path(labels_dir).glob("*.txt"):
            if prefix not in label_file.name:
                continue
            with open(label_file) as f:
                lines = f.readlines()
            dup_stripped = dup_line.strip()
            found_first = False
            new_lines = []
            for line in lines:
                if line.strip() == dup_stripped and not found_first:
                    found_first = True
                    new_lines.append(line)  # keep first occurrence
                elif line.strip() == dup_stripped and found_first:
                    total_removed += 1  # skip duplicate
                else:
                    new_lines.append(line)
            if found_first:
                with open(label_file, "w") as f:
                    f.writelines(new_lines)
    return total_removed

dup_count = remove_duplicate_lines(TRAIN_LABELS, DUPLICATE_FIXES)
print(f"Fix 4 - Duplicate labels: removed {dup_count} duplicates")


# ---- Fix 5: Adjust inaccurate bounding boxes (24 files, ~25 boxes) ----

# Each entry: (filename_prefix, old_line, new_line) — coordinates refined
INACCURATE_BOX_FIXES = [
    ("MVI_40714_img00074", "1 0.41640625 0.73359375 0.084375 0.1484375",
                           "1 0.419000 0.765000 0.086000 0.120000"),
    ("MVI_40714_img00074", "1 0.26171875 0.54921875 0.06796875 0.10703125",
                           "1 0.258000 0.570000 0.060000 0.085000"),
    ("MVI_40714_img00075", "1 0.26171875 0.54921875 0.06796875 0.10703125",
                           "1 0.255000 0.573000 0.063000 0.083000"),
    ("MVI_40714_img00128", "1 0.41640625 0.73359375 0.084375 0.1484375",
                           "1 0.420000 0.765000 0.088000 0.120000"),
    ("MVI_40714_img00144", "1 0.41640625 0.73359375 0.084375 0.1484375",
                           "1 0.421000 0.763000 0.092000 0.125000"),
    ("MVI_40714_img00167", "1 0.41640625 0.73359375 0.084375 0.1484375",
                           "1 0.417000 0.765000 0.084000 0.115000"),
    ("MVI_40853_img01184", "1 0.51171875 0.53984375 0.11875 0.13671875",
                           "1 0.500000 0.556000 0.090000 0.108000"),
    ("MVI_40853_img01256", "1 0.6109375 0.4234375 0.09609375 0.1203125",
                           "1 0.602000 0.441000 0.068000 0.085000"),
    ("MVI_40853_img01297", "1 0.61015625 0.4234375 0.09609375 0.1203125",
                           "1 0.600000 0.442000 0.070000 0.085000"),
    ("MVI_40855_img00002", "3 0.48203125 0.30859375 0.19765625 0.209375",
                           "3 0.460000 0.313000 0.175000 0.185000"),
    ("MVI_40855_img00145", "1 0.35078125 0.4640625 0.11484375 0.11640625",
                           "1 0.334000 0.478000 0.075000 0.090000"),
    ("MVI_40855_img00175", "3 0.48203125 0.3265625 0.20078125 0.221875",
                           "3 0.462000 0.330000 0.182000 0.192000"),
    ("MVI_40855_img00287", "3 0.48359375 0.32109375 0.203125 0.2109375",
                           "3 0.452000 0.338000 0.180000 0.180000"),
    ("MVI_40855_img00387", "3 0.44140625 0.35078125 0.21953125 0.23359375",
                           "3 0.415000 0.368000 0.195000 0.200000"),
    ("MVI_40855_img00406", "3 0.41484375 0.37109375 0.23984375 0.2515625",
                           "3 0.385000 0.393000 0.185000 0.215000"),
    ("MVI_40855_img00425", "3 0.37265625 0.396875 0.26171875 0.2703125",
                           "3 0.340000 0.418000 0.190000 0.200000"),
    ("MVI_40855_img00483", "1 0.54375 0.478125 0.09921875 0.15",
                           "1 0.537000 0.498000 0.082000 0.112000"),
    ("MVI_40855_img00533", "1 0.62578125 0.3984375 0.09296875 0.11640625",
                           "1 0.615000 0.409000 0.070000 0.090000"),
    ("MVI_40855_img00541", "1 0.6109375 0.41484375 0.09375 0.121875",
                           "1 0.602000 0.430000 0.070000 0.093000"),
    ("MVI_40855_img00579", "1 0.52265625 0.52890625 0.11875 0.17421875",
                           "1 0.511000 0.550000 0.090000 0.128000"),
    ("MVI_40864_img01029", "3 0.434375 0.13046875 0.19296875 0.121875",
                           "3 0.450000 0.134000 0.195000 0.160000"),
    ("MVI_40864_img01372", "3 0.665625 0.43515625 0.47890625 0.3796875",
                           "3 0.655000 0.430000 0.270000 0.280000"),
    ("MVI_40864_img01372", "3 0.79140625 0.1453125 0.31953125 0.178125",
                           "3 0.860000 0.133000 0.195000 0.190000"),
    ("MVI_40903_img00596", "3 0.609375 0.134375 0.28515625 0.11875",
                           "3 0.545000 0.142000 0.185000 0.150000"),
]

inacc_count = apply_line_replacements(TRAIN_LABELS, INACCURATE_BOX_FIXES)
print(f"Fix 5 - Inaccurate boxes: refined {inacc_count} bounding boxes")


# ---- Fix 6: Scene-specific fixes ----

# MVI_40863_img01651: remove 4 lingering stuck labels (same coords as Fix 2a,
# but this frame was in the "scene review" category rather than the batch fix)
def scene_fix_40863_01651(labels_dir):
    """Remove remaining stuck labels from MVI_40863 frame 01651."""
    stuck_lines = {
        "2 0.78671875 0.4890625 0.2 0.228125",
        "1 0.55078125 0.42890625 0.1625 0.14609375",
        "2 0.396875 0.36015625 0.11796875 0.140625",
        "1 0.27265625 0.33984375 0.10546875 0.1015625",
    }
    removed = 0
    for label_file in Path(labels_dir).glob("*.txt"):
        if "MVI_40863_img01651" not in label_file.name:
            continue
        with open(label_file) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.strip() in stuck_lines:
                removed += 1
            else:
                new_lines.append(line)
        if removed > 0:
            with open(label_file, "w") as f:
                f.writelines(new_lines)
    return removed

# MVI_40905_img00027: add missing bus annotation
def scene_fix_40905_00027(labels_dir):
    """Add a missing bus annotation to MVI_40905 frame 00027."""
    added = 0
    for label_file in Path(labels_dir).glob("*.txt"):
        if "MVI_40905_img00027" not in label_file.name:
            continue
        with open(label_file) as f:
            content = f.read()
        missing_line = "3 0.215 0.115 0.12 0.065"
        if missing_line not in content:
            # Ensure file ends with newline before appending
            if content and not content.endswith("\n"):
                content += "\n"
            content += missing_line + "\n"
            with open(label_file, "w") as f:
                f.write(content)
            added += 1
    return added

r1 = scene_fix_40863_01651(TRAIN_LABELS)
r2 = scene_fix_40905_00027(TRAIN_LABELS)
print(f"Fix 6 - Scene-specific: removed {r1} stuck labels from MVI_40863_01651, "
      f"added {r2} missing bus annotation to MVI_40905_00027")

# ---- Summary ----
after_train = count_labels(TRAIN_LABELS)
after_val = count_labels(VAL_LABELS)
print(f"\nLabel cleaning complete.")
print(f"  Train: {before_train} -> {after_train} labels ({before_train - after_train} net removed)")
print(f"  Val:   {before_val} -> {after_val} labels ({before_val - after_val} net removed)")

# ---- Fix 7: Remove phantom single-frame labels (tiny car boxes that appear in only one frame) ----
# These are annotation noise — tiny boxes that don't persist across frames
print("Fix 7 - Removing phantom single-frame tiny labels...")
phantom_removed = 0
phantom_files = 0

# Group labels by scene
from collections import defaultdict
import re

scene_files = defaultdict(list)
for f in sorted(TRAIN_LABELS.glob("*.txt")):
    m = re.match(r"(MVI_\d+)", f.name)
    if m:
        scene_files[m.group(1)].append(f)

for scene, files in scene_files.items():
    if len(files) < 3:
        continue
    # Sort by frame number
    def frame_num(f):
        m = re.search(r'img(\d+)', f.name)
        return int(m.group(1)) if m else 0
    files.sort(key=frame_num)

    # For each file, check if any label is "phantom" (tiny + not in neighbors)
    for i, f in enumerate(files):
        lines = f.read_text().strip().split("\n") if f.read_text().strip() else []
        if not lines:
            continue

        # Get neighbor labels
        neighbor_lines = set()
        for j in [i-1, i+1]:
            if 0 <= j < len(files):
                nl = files[j].read_text().strip().split("\n") if files[j].read_text().strip() else []
                neighbor_lines.update(nl)

        kept = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                kept.append(line)
                continue
            cls, x, y, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            area = w * h

            # Only remove if: tiny (<0.5% area) AND car/van (class 1 or 2) AND no match in neighbors
            if area < 0.005 and cls in (1, 2):
                # Check if similar box exists in neighbors (within 5% position tolerance)
                has_match = False
                for nl in neighbor_lines:
                    np_ = nl.strip().split()
                    if len(np_) >= 5:
                        nc, nx, ny = int(np_[0]), float(np_[1]), float(np_[2])
                        if nc == cls and abs(nx - x) < 0.05 and abs(ny - y) < 0.05:
                            has_match = True
                            break
                if not has_match:
                    phantom_removed += 1
                    continue  # skip this line
            kept.append(line)

        if len(kept) < len(lines):
            f.write_text("\n".join(kept) + ("\n" if kept else ""))
            phantom_files += 1

print(f"  Removed {phantom_removed} phantom tiny labels from {phantom_files} files")

after_phantom = count_labels(TRAIN_LABELS)
print(f"  Train labels now: {after_phantom}")


# ---- Fix 8: Cycle 2 label fixes (129 missing labels + 31 wrong class corrections) ----
# Identified by vision-based review of model predictions vs labels.
# Missing labels: vehicles visible in images but not annotated.
# Wrong class: class ID errors (e.g., van labeled as car, car labeled as van).

print("Fix 8 - Applying cycle 2 label corrections...")

def find_label_file(labels_dir, prefix):
    """Find a label file matching the MVI_XXXXX_imgYYYYY prefix (hash may differ on Kaggle)."""
    for f in Path(labels_dir).glob("*.txt"):
        if prefix in f.name:
            return f
    return None

def append_lines_to_file(labels_dir, prefix, new_lines):
    """Append label lines to a file, matching by prefix."""
    f = find_label_file(labels_dir, prefix)
    if f is None:
        return 0
    content = f.read_text()
    if content and not content.endswith("\n"):
        content += "\n"
    added = 0
    for line in new_lines:
        if line.strip() not in content:
            content += line.strip() + "\n"
            added += 1
    f.write_text(content)
    return added

def replace_class_in_file(labels_dir, prefix, old_line, new_line):
    """Replace a specific label line (class correction) matching by prefix."""
    f = find_label_file(labels_dir, prefix)
    if f is None:
        return 0
    content = f.read_text()
    old_stripped = old_line.strip()
    if old_stripped not in content:
        return 0
    lines = content.splitlines()
    new_lines_out = []
    replaced = 0
    for ln in lines:
        if ln.strip() == old_stripped:
            new_lines_out.append(new_line.strip())
            replaced += 1
        else:
            new_lines_out.append(ln)
    f.write_text("\n".join(new_lines_out) + "\n")
    return replaced

# --- 8a: Add 129 missing labels across 10 files ---

CYCLE2_MISSING_LABELS = {
    "MVI_40863_img01542": [
        "1 0.553 0.428 0.168178 0.154722",
        "2 0.789 0.490 0.216696 0.243138",
        "2 0.394 0.357 0.124865 0.143711",
        "1 0.271 0.337 0.11153 0.104236",
        "1 0.843 0.153 0.119759 0.101453",
        "1 0.471 0.159 0.072129 0.069101",
        "1 0.695 0.142 0.103307 0.089039",
        "1 0.935 0.204 0.110653 0.099322",
        "1 0.491 0.134 0.065516 0.058247",
        "1 0.569 0.154 0.117309 0.092647",
        "1 0.018 0.192 0.034404 0.046077",
        "1 0.420 0.125 0.064447 0.059785",
        "1 0.967 0.050 0.066058 0.099431",
    ],
    "MVI_40863_img01546": [
        "2 0.790 0.491 0.219679 0.242498",
        "1 0.553 0.430 0.171739 0.15846",
        "1 0.272 0.337 0.111165 0.103989",
        "2 0.395 0.361 0.12668 0.151224",
        "1 0.473 0.157 0.069411 0.066567",
        "1 0.493 0.135 0.063848 0.057503",
        "1 0.847 0.155 0.125516 0.108655",
        "1 0.696 0.143 0.107078 0.091408",
        "1 0.571 0.153 0.112886 0.089387",
        "1 0.939 0.209 0.096218 0.106602",
        "1 0.018 0.192 0.033742 0.046674",
    ],
    "MVI_40863_img01549": [
        "1 0.553 0.430 0.169343 0.156757",
        "2 0.790 0.491 0.21764 0.241108",
        "1 0.271 0.337 0.111897 0.103633",
        "2 0.395 0.360 0.125445 0.149272",
        "1 0.950 0.210 0.098758 0.110724",
        "1 0.852 0.153 0.124799 0.102599",
        "1 0.473 0.157 0.068946 0.066041",
        "1 0.494 0.134 0.067162 0.060588",
        "1 0.698 0.140 0.108317 0.088496",
        "1 0.975 0.600 0.05091 0.200723",
        "1 0.568 0.154 0.101396 0.08375",
        "1 0.018 0.192 0.03433 0.046562",
        "1 0.081 0.197 0.065011 0.062929",
    ],
    "MVI_40863_img01609": [
        "1 0.553 0.431 0.167137 0.158489",
        "2 0.789 0.490 0.21638 0.245335",
        "1 0.272 0.338 0.111314 0.106798",
        "2 0.394 0.362 0.124976 0.149719",
        "1 0.619 0.164 0.114 0.096954",
        "1 0.780 0.190 0.132594 0.099265",
        "1 0.975 0.592 0.050644 0.178123",
        "1 0.749 0.151 0.120343 0.097871",
        "1 0.968 0.785 0.064961 0.267543",
        "1 0.957 0.164 0.086968 0.105697",
        "1 0.313 0.150 0.056974 0.055017",
        "1 0.349 0.130 0.06165 0.05924",
        "1 0.017 0.194 0.033361 0.049792",
    ],
    "MVI_40863_img01620": [
        "1 0.553 0.429 0.1674 0.154323",
        "2 0.789 0.492 0.215646 0.243959",
        "2 0.394 0.359 0.125219 0.14079",
        "1 0.271 0.338 0.112052 0.107424",
        "1 0.642 0.167 0.105262 0.091586",
        "1 0.974 0.600 0.051695 0.197822",
        "1 0.805 0.193 0.140311 0.102837",
        "1 0.317 0.150 0.053719 0.054443",
        "1 0.977 0.083 0.045136 0.081406",
        "1 0.351 0.128 0.06063 0.053985",
        "1 0.761 0.153 0.112963 0.096112",
        "1 0.975 0.163 0.049007 0.099314",
        "1 0.373 0.157 0.051942 0.055133",
        "1 0.017 0.192 0.033842 0.048045",
    ],
    "MVI_40863_img01621": [
        "1 0.553 0.429 0.165904 0.154884",
        "2 0.789 0.491 0.214518 0.243455",
        "2 0.394 0.359 0.124919 0.139711",
        "1 0.271 0.338 0.111809 0.106444",
        "1 0.646 0.168 0.102845 0.090804",
        "1 0.974 0.601 0.05109 0.197903",
        "1 0.315 0.151 0.055424 0.05418",
        "1 0.351 0.128 0.060229 0.052363",
        "1 0.762 0.153 0.109622 0.096321",
        "1 0.806 0.193 0.139358 0.105374",
        "1 0.975 0.083 0.050862 0.080666",
        "1 0.978 0.166 0.044386 0.101381",
        "1 0.375 0.157 0.05691 0.055263",
        "1 0.017 0.193 0.033879 0.04736",
        "1 0.288 0.129 0.049423 0.050596",
    ],
    "MVI_40863_img01651": [
        "1 0.551 0.431 0.16717 0.156411",
        "2 0.790 0.491 0.215939 0.242929",
        "2 0.394 0.364 0.129802 0.149685",
        "1 0.389 0.155 0.068818 0.061421",
        "1 0.884 0.202 0.143343 0.102855",
        "1 0.973 0.597 0.054273 0.197819",
        "1 0.268 0.338 0.117728 0.110945",
        "1 0.420 0.132 0.064514 0.057386",
        "1 0.808 0.160 0.125424 0.101782",
        "1 0.934 0.121 0.131009 0.099915",
        "1 0.013 0.292 0.02555 0.096597",
        "1 0.017 0.193 0.032662 0.046731",
        "1 0.353 0.133 0.066098 0.060712",
    ],
    "MVI_40902_img00846": [
        "1 0.128 0.268 0.118139 0.092058",
        "1 0.050 0.324 0.098135 0.097683",
        "1 0.133 0.210 0.105698 0.077837",
        "1 0.176 0.180 0.096174 0.074381",
        "1 0.057 0.193 0.09437 0.074439",
        "1 0.189 0.143 0.083152 0.062983",
        "1 0.971 0.177 0.057294 0.085903",
        "0 0.597 0.626 0.296524 0.266181",
        "1 0.008 0.229 0.016018 0.076762",
        "1 0.419 0.912 0.097364 0.099539",
    ],
    "MVI_40904_img00606": [
        "1 0.895 0.148 0.071818 0.054121",
        "1 0.066 0.362 0.104019 0.110626",
        "1 0.045 0.324 0.088584 0.100357",
        "3 0.218 0.119 0.104261 0.148649",
        "1 0.979 0.132 0.041224 0.053776",
        "2 0.139 0.222 0.061915 0.090386",
        "1 0.057 0.260 0.075826 0.075208",
        "0 0.593 0.654 0.30407 0.268258",
        "3 0.143 0.136 0.121868 0.125057",
        "1 0.864 0.905 0.271163 0.189",
        "1 0.932 0.124 0.049143 0.053434",
        "1 0.981 0.100 0.038068 0.043818",
        "1 0.017 0.276 0.033389 0.096365",
        "1 0.459 0.911 0.304342 0.177316",
    ],
    "MVI_40904_img01163": [
        "1 0.090 0.306 0.090161 0.095956",
        "1 0.960 0.150 0.056235 0.052336",
        "3 0.091 0.167 0.153927 0.136716",
        "1 0.327 0.225 0.092547 0.078691",
        "1 0.044 0.911 0.088594 0.1779",
        "1 0.141 0.201 0.06198 0.066297",
        "1 0.974 0.105 0.051754 0.04837",
        "1 0.372 0.203 0.071273 0.062864",
        "2 0.066 0.245 0.06792 0.092591",
        "0 0.595 0.651 0.302894 0.277117",
        "1 0.270 0.909 0.299048 0.182836",
        "1 0.684 0.903 0.315941 0.194071",
        "1 0.952 0.921 0.095518 0.157755",
    ],
}

c2_added = 0
c2_add_files = 0
for prefix, lines in CYCLE2_MISSING_LABELS.items():
    n = append_lines_to_file(TRAIN_LABELS, prefix, lines)
    if n > 0:
        c2_added += n
        c2_add_files += 1
print(f"  Fix 8a - Missing labels: added {c2_added} labels to {c2_add_files} files")

# --- 8b: Correct 28 wrong class IDs across 24 files ---
# Each entry: (filename_prefix, old_line, new_line)

CYCLE2_WRONG_CLASS_FIXES = [
    # van (2) -> car (1): MVI_40131
    ("MVI_40131_img01508", "2 0.278125 0.18046875 0.078125 0.0890625",
                           "1 0.278125 0.18046875 0.078125 0.0890625"),
    # van (2) -> car (1): MVI_40192
    ("MVI_40192_img01538", "2 0.82734375 0.915625 0.165625 0.1484375",
                           "1 0.82734375 0.915625 0.165625 0.1484375"),
    ("MVI_40192_img01557", "2 0.734375 0.56953125 0.10078125 0.17578125",
                           "1 0.734375 0.56953125 0.10078125 0.17578125"),
    ("MVI_40192_img01562", "2 0.71953125 0.509375 0.09296875 0.1609375",
                           "1 0.71953125 0.509375 0.09296875 0.1609375"),
    ("MVI_40192_img01613", "2 0.65703125 0.2203125 0.040625 0.078125",
                           "1 0.65703125 0.2203125 0.040625 0.078125"),
    # van (2) -> car (1): MVI_40201
    ("MVI_40201_img00157", "2 0.17109375 0.7546875 0.15078125 0.3390625",
                           "1 0.17109375 0.7546875 0.15078125 0.3390625"),
    # van (2) -> car (1): MVI_40775
    ("MVI_40775_img00075", "2 0.30625 0.15078125 0.05078125 0.0890625",
                           "1 0.30625 0.15078125 0.05078125 0.0890625"),
    # van (2) -> car (1): MVI_40864
    ("MVI_40864_img00529", "2 0.8375 0.415625 0.221875 0.20703125",
                           "1 0.8375 0.415625 0.221875 0.20703125"),
    # van (2) -> car (1): MVI_40871
    ("MVI_40871_img00162", "2 0.15859375 0.31640625 0.05078125 0.08359375",
                           "1 0.15859375 0.31640625 0.05078125 0.08359375"),
    ("MVI_40871_img00185", "2 0.4328125 0.22578125 0.06484375 0.06484375",
                           "1 0.4328125 0.22578125 0.06484375 0.06484375"),
    ("MVI_40871_img00185", "2 0.18671875 0.35703125 0.06484375 0.1015625",
                           "1 0.18671875 0.35703125 0.06484375 0.1015625"),
    ("MVI_40871_img00333", "2 0.4984375 0.2359375 0.07421875 0.07578125",
                           "1 0.4984375 0.2359375 0.07421875 0.07578125"),
    # bus (3) -> truck (0): MVI_40902_img00414
    ("MVI_40902_img00414", "3 0.8953125 0.32421875 0.209375 0.26640625",
                           "0 0.8953125 0.32421875 0.209375 0.26640625"),
    # van (2) -> car (1): MVI_40902_img00414
    ("MVI_40902_img00414", "2 0.85859375 0.128125 0.07578125 0.06328125",
                           "1 0.85859375 0.128125 0.07578125 0.06328125"),
    # car (1) -> van (2): MVI_40902_img00591
    ("MVI_40902_img00591", "1 0.31953125 0.31015625 0.115625 0.1015625",
                           "2 0.31953125 0.31015625 0.115625 0.1015625"),
    # van (2) -> car (1): MVI_40902_img00846 (from missing-labels batch)
    ("MVI_40902_img00846", "2 0.89921875 0.128125 0.09375 0.06875",
                           "1 0.89921875 0.128125 0.09375 0.06875"),
    # van (2) -> car (1): MVI_40902_img00954
    ("MVI_40902_img00954", "2 0.36875 0.13046875 0.08515625 0.06484375",
                           "1 0.36875 0.13046875 0.08515625 0.06484375"),
    # van (2) -> car (1): MVI_40902_img00954
    ("MVI_40902_img00954", "2 0.496875 0.1 0.06640625 0.05703125",
                           "1 0.496875 0.1 0.06640625 0.05703125"),
    # car (1) -> truck (0): MVI_40902_img00954
    ("MVI_40902_img00954", "1 0.26328125 0.28046875 0.18203125 0.16328125",
                           "0 0.26328125 0.28046875 0.18203125 0.16328125"),
    # car (1) -> van (2): MVI_40902_img00977
    ("MVI_40902_img00977", "1 0.96328125 0.26015625 0.07265625 0.08359375",
                           "2 0.96328125 0.26015625 0.07265625 0.08359375"),
    # van (2) -> car (1): MVI_40904_img00499
    ("MVI_40904_img00499", "2 0.92421875 0.3484375 0.15078125 0.11640625",
                           "1 0.92421875 0.3484375 0.15078125 0.11640625"),
    # van (2) -> car (1): MVI_40904_img00539
    ("MVI_40904_img00539", "2 0.7421875 0.28828125 0.125 0.10390625",
                           "1 0.7421875 0.28828125 0.125 0.10390625"),
    # van (2) -> car (1): MVI_40904_img00540
    ("MVI_40904_img00540", "2 0.73828125 0.28671875 0.125 0.10390625",
                           "1 0.73828125 0.28671875 0.125 0.10390625"),
    # van (2) -> car (1): MVI_40904_img00551
    ("MVI_40904_img00551", "2 0.69140625 0.2796875 0.12421875 0.1",
                           "1 0.69140625 0.2796875 0.12421875 0.1"),
    # van (2) -> car (1): MVI_40904_img00552
    ("MVI_40904_img00552", "2 0.6875 0.278125 0.12265625 0.1",
                           "1 0.6875 0.278125 0.12265625 0.1"),
    # van (2) -> car (1): MVI_40904_img00599
    ("MVI_40904_img00599", "2 0.528125 0.25 0.1125 0.09296875",
                           "1 0.528125 0.25 0.1125 0.09296875"),
    # van (2) -> car (1): MVI_40904_img00606 (from missing-labels batch)
    ("MVI_40904_img00606", "2 0.5078125 0.24921875 0.11171875 0.09453125",
                           "1 0.5078125 0.24921875 0.11171875 0.09453125"),
    # van (2) -> car (1): MVI_40904_img00622
    ("MVI_40904_img00622", "2 0.46328125 0.240625 0.109375 0.0890625",
                           "1 0.46328125 0.240625 0.109375 0.0890625"),
    # van (2) -> car (1): MVI_40904_img00675
    ("MVI_40904_img00675", "2 0.36015625 0.2296875 0.10234375 0.08359375",
                           "1 0.36015625 0.2296875 0.10234375 0.08359375"),
    # van (2) -> car (1): MVI_40904_img00706
    ("MVI_40904_img00706", "2 0.32890625 0.2265625 0.08984375 0.0796875",
                           "1 0.32890625 0.2265625 0.08984375 0.0796875"),
]

c2_class_fixed = 0
for prefix, old_line, new_line in CYCLE2_WRONG_CLASS_FIXES:
    c2_class_fixed += replace_class_in_file(TRAIN_LABELS, prefix, old_line, new_line)
print(f"  Fix 8b - Wrong class: corrected {c2_class_fixed} labels across {len(CYCLE2_WRONG_CLASS_FIXES)} entries")

after_c2 = count_labels(TRAIN_LABELS)
print(f"  Train labels after cycle 2 fixes: {after_c2}")


# ---- Fix 9: Cycle 3 label fixes (132 missing labels + 22 wrong class corrections) ----
# Identified by continued vision-based review of model predictions vs labels after cycle 2.
# Missing labels: vehicles visible in images but not annotated.
# Wrong class: class ID errors.

print("Fix 9 - Applying cycle 3 label corrections...")

# --- 9a: Add 132 missing labels across 15 files ---

CYCLE3_MISSING_LABELS = {
    "MVI_40192_img01557": [
        "1 0.240000 0.899000 0.191941 0.202419",
        "1 0.449000 0.174000 0.035825 0.052390",
        "1 0.636000 0.167000 0.030485 0.049845",
        "1 0.514000 0.168000 0.032537 0.052264",
        "1 0.819000 0.156000 0.038675 0.053500",
        "1 0.744000 0.140000 0.032817 0.046930",
        "1 0.681000 0.135000 0.031284 0.043413",
        "1 0.489000 0.133000 0.030971 0.053263",
    ],
    "MVI_40201_img00157": [
        "1 0.195000 0.119000 0.042849 0.053998",
        "1 0.359000 0.098000 0.050020 0.049051",
        "1 0.398000 0.081000 0.042016 0.045708",
        "1 0.298000 0.095000 0.037516 0.042442",
        "1 0.095000 0.094000 0.031946 0.046562",
        "1 0.014000 0.143000 0.027909 0.056018",
        "1 0.259000 0.054000 0.032201 0.040151",
        "1 0.371000 0.054000 0.035448 0.039578",
    ],
    "MVI_40201_img00170": [
        "1 0.408000 0.089000 0.045099 0.050557",
        "1 0.323000 0.082000 0.047113 0.048197",
        "1 0.368000 0.105000 0.045982 0.050982",
        "1 0.106000 0.109000 0.034423 0.049685",
        "1 0.442000 0.078000 0.044107 0.046043",
        "1 0.340000 0.110000 0.045110 0.049498",
        "1 0.018000 0.172000 0.035429 0.060452",
        "1 0.364000 0.067000 0.037643 0.038133",
        "1 0.266000 0.082000 0.036051 0.041609",
        "1 0.603000 0.972000 0.240653 0.056941",
    ],
    "MVI_40201_img00176": [
        "1 0.390000 0.083000 0.042900 0.046723",
        "1 0.112000 0.117000 0.036630 0.051976",
        "1 0.424000 0.071000 0.040059 0.043744",
        "1 0.350000 0.096000 0.039723 0.047879",
        "1 0.322000 0.101000 0.045056 0.045554",
        "1 0.350000 0.061000 0.033366 0.034695",
        "1 0.307000 0.076000 0.043267 0.045542",
        "1 0.040000 0.097000 0.030472 0.041489",
        "1 0.106000 0.058000 0.030229 0.043099",
        "1 0.253000 0.076000 0.038799 0.038201",
    ],
    "MVI_40201_img00183": [
        "1 0.369000 0.068000 0.039036 0.044978",
        "1 0.118000 0.120000 0.036838 0.053368",
        "1 0.403000 0.060000 0.042237 0.042847",
        "1 0.329000 0.081000 0.039966 0.044455",
        "1 0.301000 0.087000 0.041773 0.042357",
        "1 0.335000 0.048000 0.032741 0.033195",
        "1 0.289000 0.062000 0.039350 0.040905",
        "1 0.971000 0.505000 0.057688 0.141324",
        "1 0.111000 0.057000 0.028440 0.043073",
        "1 0.043000 0.097000 0.032316 0.045022",
    ],
    "MVI_40201_img00229": [
        "1 0.161000 0.118000 0.040615 0.058004",
        "1 0.098000 0.119000 0.034627 0.047806",
        "1 0.307000 0.086000 0.038728 0.042241",
        "1 0.050000 0.117000 0.033720 0.050202",
        "1 0.105000 0.068000 0.026552 0.035510",
        "1 0.294000 0.108000 0.043637 0.044499",
    ],
    "MVI_40201_img00459": [
        "1 0.961000 0.343000 0.078206 0.137098",
        "1 0.131000 0.151000 0.043227 0.060545",
        "1 0.343000 0.103000 0.038463 0.045629",
        "1 0.143000 0.111000 0.034104 0.040293",
        "1 0.291000 0.096000 0.035877 0.042231",
        "1 0.379000 0.085000 0.034299 0.041574",
        "1 0.240000 0.098000 0.037498 0.037594",
        "2 0.322000 0.078000 0.042097 0.051022",
    ],
    "MVI_40201_img00774": [
        "1 0.217000 0.141000 0.042036 0.057319",
        "2 0.160000 0.142000 0.052203 0.093985",
        "1 0.104000 0.101000 0.028903 0.038777",
        "1 0.285000 0.074000 0.030065 0.037008",
        "1 0.147000 0.088000 0.030452 0.041722",
        "1 0.112000 0.062000 0.027673 0.038117",
        "1 0.079000 0.073000 0.027286 0.037615",
        "1 0.268000 0.092000 0.034778 0.038974",
        "1 0.334000 0.079000 0.040431 0.044199",
        "1 0.360000 0.061000 0.032190 0.036643",
    ],
    "MVI_40863_img01548": [
        "1 0.550986 0.429000 0.171439 0.155779",
        "2 0.395000 0.361000 0.126792 0.155900",
        "2 0.786000 0.486000 0.219724 0.249183",
        "1 0.272000 0.339000 0.110329 0.109314",
        "1 0.950000 0.210000 0.098997 0.101656",
        "1 0.854000 0.153000 0.117331 0.103761",
        "1 0.468000 0.155000 0.077591 0.071120",
        "1 0.492000 0.135000 0.066576 0.059351",
        "1 0.697000 0.140000 0.104589 0.086479",
        "1 0.566000 0.156000 0.100471 0.079719",
        "2 0.973000 0.599000 0.053386 0.267419",
        "3 0.233000 0.124000 0.127514 0.111403",
    ],
    "MVI_40863_img01608": [
        "2 0.394080 0.361627 0.127789 0.154001",
        "1 0.550986 0.431110 0.166281 0.155273",
        "2 0.785603 0.487966 0.211601 0.244363",
        "1 0.271444 0.339258 0.113174 0.110023",
        "1 0.744511 0.149205 0.111769 0.091649",
        "1 0.960478 0.790763 0.077921 0.278297",
        "1 0.613650 0.163485 0.112668 0.095200",
        "1 0.974734 0.593444 0.050052 0.181072",
        "1 0.783521 0.189609 0.126684 0.093315",
        "1 0.312661 0.150736 0.058814 0.055051",
        "1 0.352667 0.130158 0.066013 0.058045",
        "1 0.954490 0.158672 0.090955 0.090710",
    ],
    "MVI_40864_img00030": [
        "1 0.215000 0.230000 0.075538 0.069012",
        "1 0.890000 0.124000 0.120156 0.084915",
        "1 0.460000 0.104000 0.072342 0.061144",
    ],
    "MVI_40902_img00977": [
        "1 0.063000 0.317000 0.120709 0.093056",
        "1 0.181000 0.180000 0.096054 0.069395",
        "1 0.056000 0.192000 0.088615 0.066725",
        "0 0.594000 0.628000 0.298773 0.264709",
        "1 0.138000 0.207000 0.093190 0.067467",
        "2 0.976000 0.181000 0.048367 0.077153",
        "2 0.228000 0.133000 0.081647 0.063731",
    ],
    "MVI_40904_img01160": [
        "1 0.684000 0.904000 0.307610 0.188233",
        "1 0.969000 0.157000 0.060014 0.060037",
        "1 0.043000 0.914000 0.085113 0.172664",
        "1 0.270000 0.910000 0.291445 0.175303",
        "0 0.591000 0.655000 0.303000 0.267712",
        "1 0.951000 0.921000 0.097483 0.157780",
        "1 0.324000 0.224000 0.094551 0.077713",
        "2 0.068000 0.243000 0.071073 0.089822",
        "1 0.368000 0.205000 0.076225 0.066887",
        "1 0.095000 0.300000 0.089526 0.099394",
    ],
    "MVI_40904_img01162": [
        "1 0.684000 0.905000 0.307980 0.187728",
        "1 0.270000 0.911000 0.292113 0.174815",
        "1 0.043000 0.913000 0.085332 0.173188",
        "0 0.591000 0.654000 0.303024 0.267835",
        "1 0.325000 0.224000 0.094213 0.076520",
        "1 0.951000 0.922000 0.097666 0.156800",
        "1 0.090000 0.305000 0.088782 0.098818",
        "2 0.068000 0.243000 0.071948 0.089217",
        "1 0.964000 0.152000 0.063500 0.057928",
        "1 0.369000 0.205000 0.076679 0.067346",
    ],
    "MVI_40904_img01257": [
        "1 0.685000 0.905000 0.307923 0.187589",
        "1 0.043000 0.913000 0.085637 0.174038",
        "1 0.271000 0.910000 0.291540 0.175053",
        "0 0.593000 0.654000 0.303693 0.268196",
        "1 0.325000 0.225000 0.092885 0.079180",
        "1 0.370000 0.204000 0.075157 0.067513",
        "1 0.951000 0.924000 0.097328 0.152541",
        "3 0.049000 0.212000 0.098783 0.174793",
    ],
}

c3_added = 0
c3_add_files = 0
for prefix, new_lines in CYCLE3_MISSING_LABELS.items():
    n = append_lines_to_file(TRAIN_LABELS, prefix, new_lines)
    if n > 0:
        c3_added += n
        c3_add_files += 1
print(f"  Fix 9a - Missing labels: added {c3_added} labels to {c3_add_files} files")

# --- 9b: Correct 22 wrong class IDs ---
# Each entry: (filename_prefix, old_line, new_line)

CYCLE3_WRONG_CLASS_FIXES = [
    # van (2) -> car (1): MVI_40131_img01508
    ("MVI_40131_img01508", "2 0.73828125 0.921875 0.18984375 0.13125",
                           "1 0.73828125 0.921875 0.18984375 0.13125"),
    # van (2) -> car (1): MVI_40201_img00099
    ("MVI_40201_img00099", "2 0.03125 0.2640625 0.06015625 0.1109375",
                           "1 0.03125 0.2640625 0.06015625 0.1109375"),
    # car (1) -> van (2): MVI_40201_img00157 (revert cycle 2 Fix 8b)
    ("MVI_40201_img00157", "1 0.17109375 0.7546875 0.15078125 0.3390625",
                           "2 0.17109375 0.7546875 0.15078125 0.3390625"),
    # car (1) -> van (2): MVI_40771_img00162
    ("MVI_40771_img00162", "1 0.07578125 0.57109375 0.14921875 0.28359375",
                           "2 0.07578125 0.57109375 0.14921875 0.28359375"),
    # car (1) -> van (2): MVI_40864_img00529 (revert cycle 2 Fix 8b)
    ("MVI_40864_img00529", "1 0.8375 0.415625 0.221875 0.20703125",
                           "2 0.8375 0.415625 0.221875 0.20703125"),
    # car (1) -> van (2): MVI_40892_img01778
    ("MVI_40892_img01778", "1 0.590625 0.5640625 0.15390625 0.2015625",
                           "2 0.590625 0.5640625 0.15390625 0.2015625"),
    # car (1) -> van (2): MVI_40902_img00208
    ("MVI_40902_img00208", "1 0.33046875 0.29140625 0.11796875 0.10390625",
                           "2 0.33046875 0.29140625 0.11796875 0.10390625"),
    # truck (0) -> bus (3): MVI_40902_img00414 (revert cycle 2 Fix 8b)
    ("MVI_40902_img00414", "0 0.8953125 0.32421875 0.209375 0.26640625",
                           "3 0.8953125 0.32421875 0.209375 0.26640625"),
    # car (1) -> van (2): MVI_40902_img00414 (revert cycle 2 Fix 8b)
    ("MVI_40902_img00414", "1 0.85859375 0.128125 0.07578125 0.06328125",
                           "2 0.85859375 0.128125 0.07578125 0.06328125"),
    # van (2) -> car (1): MVI_40902_img00930
    ("MVI_40902_img00930", "2 0.50234375 0.12890625 0.08984375 0.06484375",
                           "1 0.50234375 0.12890625 0.08984375 0.06484375"),
    # van (2) -> car (1): MVI_40902_img00944
    ("MVI_40902_img00944", "2 0.42578125 0.13359375 0.08828125 0.06640625",
                           "1 0.42578125 0.13359375 0.08828125 0.06640625"),
    # car (1) -> van (2): MVI_40902_img00954 (revert cycle 2 Fix 8b)
    ("MVI_40902_img00954", "1 0.36875 0.13046875 0.08515625 0.06484375",
                           "2 0.36875 0.13046875 0.08515625 0.06484375"),
    # car (1) -> van (2): MVI_40902_img00954 (revert cycle 2 Fix 8b)
    ("MVI_40902_img00954", "1 0.496875 0.1 0.06640625 0.05703125",
                           "2 0.496875 0.1 0.06640625 0.05703125"),
    # van (2) -> car (1): MVI_40902_img00977 (revert cycle 2 Fix 8b)
    ("MVI_40902_img00977", "2 0.96328125 0.26015625 0.07265625 0.08359375",
                           "1 0.96328125 0.26015625 0.07265625 0.08359375"),
    # car (1) -> van (2): MVI_40903_img00465
    ("MVI_40903_img00465", "1 0.52890625 0.26171875 0.11875 0.10546875",
                           "2 0.52890625 0.26171875 0.11875 0.10546875"),
    # car (1) -> van (2): MVI_40903_img00465
    ("MVI_40903_img00465", "1 0.87734375 0.16640625 0.08671875 0.0796875",
                           "2 0.87734375 0.16640625 0.08671875 0.0796875"),
    # car (1) -> van (2): MVI_40903_img00467
    ("MVI_40903_img00467", "1 0.871875 0.16953125 0.08671875 0.0796875",
                           "2 0.871875 0.16953125 0.08671875 0.0796875"),
    # car (1) -> van (2): MVI_40903_img01002
    ("MVI_40903_img01002", "1 0.31796875 0.37890625 0.10078125 0.0984375",
                           "2 0.31796875 0.37890625 0.10078125 0.0984375"),
    # car (1) -> bus (3): MVI_40904_img00015
    ("MVI_40904_img00015", "1 0.05078125 0.3953125 0.09921875 0.23671875",
                           "3 0.05078125 0.3953125 0.09921875 0.23671875"),
    # car (1) -> van (2): MVI_40904_img00043
    ("MVI_40904_img00043", "1 0.05390625 0.48984375 0.10625 0.13125",
                           "2 0.05390625 0.48984375 0.10625 0.13125"),
    # van (2) -> car (1): MVI_40904_img00722
    ("MVI_40904_img00722", "2 0.32421875 0.2265625 0.09296875 0.08125",
                           "1 0.32421875 0.2265625 0.09296875 0.08125"),
    # van (2) -> car (1): MVI_40904_img00738
    ("MVI_40904_img00738", "2 0.32578125 0.22578125 0.090625 0.0796875",
                           "1 0.32578125 0.22578125 0.090625 0.0796875"),
]

c3_class_fixed = 0
for prefix, old_line, new_line in CYCLE3_WRONG_CLASS_FIXES:
    c3_class_fixed += replace_class_in_file(TRAIN_LABELS, prefix, old_line, new_line)
print(f"  Fix 9b - Wrong class: corrected {c3_class_fixed} labels across {len(CYCLE3_WRONG_CLASS_FIXES)} entries")

after_c3 = count_labels(TRAIN_LABELS)
print(f"  Train labels after cycle 3 fixes: {after_c3}")


# Delete caches again after all modifications
for cache in DATA_DIR.rglob("*.cache"):
    cache.unlink()


# ============================================================================
# SECTION 4: Write dataset.yaml for YOLO training
# ============================================================================

dataset_yaml_path = WORK_DIR / "dataset.yaml"
dataset_yaml_content = f"""# YOLO dataset config — auto-generated for Kaggle
path: {WORK_DIR}
train: data/train/images
val: data/val/images
test: data/test/images

nc: 4
names:
  0: truck
  1: car
  2: van
  3: bus
"""
dataset_yaml_path.write_text(dataset_yaml_content)
print(f"Wrote {dataset_yaml_path}")


# ============================================================================
# SECTION 5: Login to 3LC (optional — for Dashboard tracking)
# ============================================================================

# Uncomment and set your API key to enable 3LC Dashboard integration.
# Without this, training still works but metrics won't appear in the Dashboard.

# import tlc
# tlc.login(api_key="YOUR_3LC_API_KEY_HERE")


# ============================================================================
# SECTION 6: Train YOLOv8n from scratch
# ============================================================================
#
# Hyperparameters optimized for:
#   - P100 GPU (16 GB VRAM) with batch=32
#   - Training from scratch (no pretrained weights) — needs warmer LR schedule
#   - Small vehicle detection in traffic camera images
#   - 200 epochs with cosine LR decay and strong augmentation (patience=75)

import random
import numpy as np
import torch
from ultralytics import YOLO

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("TRAINING YOLOv8n FROM SCRATCH")
print("=" * 70)
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = YOLO("yolov8n.yaml")  # architecture definition, no pretrained weights

results = model.train(
    data=str(dataset_yaml_path),
    epochs=200,
    imgsz=640,
    batch=32,
    cache="ram",
    device=0,
    workers=4,
    # Learning rate — from scratch, cosine decay to near-zero
    pretrained=False,
    lr0=0.02,
    lrf=0.001,
    cos_lr=True,
    warmup_epochs=10.0,
    warmup_momentum=0.5,
    warmup_bias_lr=0.01,
    # Augmentation — aggressive for small-object traffic scenes
    mosaic=1.0,
    mixup=0.2,
    copy_paste=0.2,
    scale=0.9,
    degrees=10.0,
    translate=0.2,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.8,
    hsv_v=0.5,
    erasing=0.3,
    close_mosaic=30,
    # Multi-scale training — critical for small vehicle detection
    multi_scale=True,
    # Regularization
    weight_decay=0.0005,
    patience=75,
    # Output
    name="kaggle_yolov8n_long",
    project=str(WORK_DIR / "runs" / "detect"),
    exist_ok=True,
    val=True,
    save_period=25,
)

WEIGHTS_PATH = WORK_DIR / "runs" / "detect" / "kaggle_yolov8n_long" / "weights" / "best.pt"
print(f"\nTraining complete. Best weights: {WEIGHTS_PATH}")


# ============================================================================
# SECTION 7: Generate predictions on test set and create submission.csv
# ============================================================================

import csv
import gc

print("=" * 70)
print("GENERATING PREDICTIONS -> SUBMISSION CSV")
print("=" * 70)

# Load best weights
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

model = YOLO(str(WEIGHTS_PATH))
print(f"  Loaded weights: {WEIGHTS_PATH}")

# Read sample submission to get expected image IDs
sample_sub = WORK_DIR / "sample_submission.csv"
rows_in = []
with open(sample_sub, newline="", encoding="utf-8") as f:
    rows_in.extend(csv.DictReader(f))

stems = [str(r["image_id"]).strip() for r in rows_in]
test_dir = DATA_DIR / "test" / "images"
print(f"  Test images: {test_dir}")
print(f"  Expected predictions: {len(stems)}")

# Find image files for each stem
def find_image(test_dir, stem):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = test_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    return None

image_paths = []
for stem in stems:
    p = find_image(test_dir, stem)
    if p is not None:
        image_paths.append(p)

print(f"  Found {len(image_paths)} / {len(stems)} test images")

# Run inference in batches
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.7
BATCH_SIZE = 32
MAX_DET = 300

pred_by_stem = {}
n = len(image_paths)
for start in range(0, n, BATCH_SIZE):
    chunk = image_paths[start:start + BATCH_SIZE]
    batch_results = model.predict(
        source=[str(p) for p in chunk],
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=0,
        max_det=MAX_DET,
        batch=min(BATCH_SIZE, len(chunk)),
        verbose=False,
    )
    for res, p in zip(batch_results, chunk):
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            pred_by_stem[p.stem] = "no box"
            continue
        xywhn = boxes.xywhn.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        order = np.argsort(-conf)
        parts = []
        for i in order:
            c = int(cls[i])
            cf = float(conf[i])
            x, y, w, h = (float(v) for v in xywhn[i])
            x = min(1.0, max(0.0, x))
            y = min(1.0, max(0.0, y))
            w = min(1.0, max(0.0, w))
            h = min(1.0, max(0.0, h))
            parts.extend([str(c), f"{cf:.6f}", f"{x:.6f}", f"{y:.6f}", f"{w:.6f}", f"{h:.6f}"])
        pred_by_stem[p.stem] = " ".join(parts)

    done = min(start + BATCH_SIZE, n)
    print(f"  {done}/{n} images processed", end="\r")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()

# Write submission CSV
submission_path = WORK_DIR / "submission.csv"
with open(submission_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["id", "image_id", "prediction_string"])
    writer.writeheader()
    for row, stem in zip(rows_in, stems):
        writer.writerow({
            "id": row["id"],
            "image_id": stem,
            "prediction_string": pred_by_stem.get(stem, "no box"),
        })

nonempty = sum(1 for stem in stems if pred_by_stem.get(stem, "no box") != "no box")
n_boxes = sum(
    len(pred_by_stem.get(stem, "").split()) // 6
    for stem in stems
    if pred_by_stem.get(stem, "no box") != "no box"
)

print("=" * 70)
print("SUBMISSION READY")
print("=" * 70)
print(f"  File: {submission_path}")
print(f"  Images with detections: {nonempty} / {len(rows_in)}")
print(f"  Total boxes: {n_boxes}")


# ============================================================================
# SECTION 8: Save weights as Kaggle output artifact
# ============================================================================

# Kaggle saves everything under /kaggle/working/ as output artifacts.
# Copy best weights to a prominent location for easy download.
output_weights = WORK_DIR / "best.pt"
if WEIGHTS_PATH.exists():
    shutil.copy2(str(WEIGHTS_PATH), str(output_weights))
    print(f"\nWeights saved to: {output_weights}")

# Also copy last.pt for potential resume
last_weights = WEIGHTS_PATH.parent / "last.pt"
if last_weights.exists():
    shutil.copy2(str(last_weights), str(WORK_DIR / "last.pt"))
    print(f"Last weights saved to: {WORK_DIR / 'last.pt'}")

print("\nDone. Submit submission.csv to the competition.")
