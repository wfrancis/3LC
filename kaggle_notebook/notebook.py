#!/usr/bin/env python3
"""
Kaggle Training Notebook — 3LC Multi-Vehicle Detection Challenge
=================================================================

Designed for Kaggle's free P100 GPU (16 GB VRAM).

This script:
  1. Installs required packages (3lc-ultralytics, umap-learn)
  2. Clones the competition repo
  3. Copies Kaggle competition data into the expected directory layout
  4. Applies all 3,074 label cleaning fixes (tiny boxes, phantom/stuck labels,
     wrong classes, duplicate labels, inaccurate bounding boxes, scene-specific
     fixes) — the core advantage of our pipeline
  5. Trains YOLOv8n from scratch with optimized hyperparameters
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

install("3lc-ultralytics")
install("umap-learn")

print("Packages installed.")


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

# Copy sample_submission.csv
sample_sub_src = KAGGLE_INPUT / "sample_submission.csv"
sample_sub_dst = WORK_DIR / "sample_submission.csv"
if sample_sub_src.exists() and not sample_sub_dst.exists():
    shutil.copy2(str(sample_sub_src), str(sample_sub_dst))
    print(f"  Copied sample_submission.csv")

print("Data setup complete.")


# ============================================================================
# SECTION 3: Apply all label cleaning fixes (3,074 total fixes)
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
#   - 120 epochs with cosine LR decay and strong augmentation

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
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

model = YOLO("yolov8n.yaml")  # architecture definition, no pretrained weights

results = model.train(
    data=str(dataset_yaml_path),
    epochs=120,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    # Learning rate — from scratch needs higher lr0 and aggressive decay
    pretrained=False,
    lr0=0.02,
    lrf=0.01,
    cos_lr=True,
    warmup_epochs=5.0,
    warmup_momentum=0.5,
    # Augmentation — strong pipeline for small-object traffic scenes
    mosaic=1.0,
    mixup=0.15,
    copy_paste=0.15,
    scale=0.9,
    degrees=5.0,
    close_mosaic=15,
    # Regularization
    weight_decay=0.0005,
    patience=30,
    # Output
    name="kaggle_yolov8n",
    project=str(WORK_DIR / "runs" / "detect"),
    exist_ok=True,
    val=True,
)

WEIGHTS_PATH = WORK_DIR / "runs" / "detect" / "kaggle_yolov8n" / "weights" / "best.pt"
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
CONF_THRESHOLD = 0.25
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
