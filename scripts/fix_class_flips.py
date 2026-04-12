#!/usr/bin/env python3
"""Fix class_flip issues found by the Rust frame-checker.

Strategy:
1. Parse report.json for all class_flip issues.
2. Group flips by scene + approximate vehicle position to identify the same
   physical vehicle across multiple frames.
3. For each vehicle track, collect the class label from ALL frames in the scene
   that contain a bbox near that position (majority vote across the full scene,
   not just the two flagged frames).
4. Overwrite minority-class frames to match the majority class.
5. Back up every modified label file before editing.

Class mapping: truck=0, car=1, van=2, bus=3
"""

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ----- Config -----
REPO = Path("/Users/william/3LC/repo")
REPORT = REPO / "rust/frame-checker/report.json"
LABELS_DIR = REPO / "data/train/labels"
BACKUP_DIR = REPO / "backups/class_flip_originals"

CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}
NAME_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

# IoU threshold for matching a label-file bbox to a flagged vehicle position
MATCH_IOU_THRESH = 0.25  # loose — vehicles drift across frames


def parse_bbox(s: str):
    """Parse 'x_center,y_center,w,h' -> tuple of floats."""
    return tuple(float(x) for x in s.split(","))


def iou(box1, box2):
    """Compute IoU between two (cx, cy, w, h) normalized boxes."""
    x1_min = box1[0] - box1[2] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def read_label_file(path: Path):
    """Read YOLO label file -> list of (class_id, cx, cy, w, h)."""
    lines = []
    if not path.exists():
        return lines
    for raw in path.read_text().strip().splitlines():
        parts = raw.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            coords = tuple(float(x) for x in parts[1:5])
            lines.append((cls, *coords))
    return lines


def read_label_file_raw(path: Path):
    """Read YOLO label file preserving raw line text -> list of (class_id, raw_line)."""
    lines = []
    if not path.exists():
        return lines
    for raw in path.read_text().strip().splitlines():
        parts = raw.strip().split()
        if len(parts) >= 5:
            cls = int(parts[0])
            lines.append((cls, raw.strip()))
    return lines


def write_label_file_raw(path: Path, raw_labels):
    """Write YOLO label file from list of (class_id, raw_line), replacing class in raw_line."""
    lines = []
    for cls, raw_line in raw_labels:
        parts = raw_line.split(None, 1)
        if len(parts) == 2:
            lines.append(f"{cls} {parts[1]}")
        else:
            lines.append(str(cls))
    path.write_text("\n".join(lines) + "\n")


def extract_frame_number(filename: str) -> int:
    """Extract frame number from filename like MVI_20012_img00352_jpg.rf.xxx.txt."""
    m = re.search(r"img(\d+)", filename)
    return int(m.group(1)) if m else -1


def extract_scene(filename: str) -> str:
    """Extract scene from filename like MVI_20012_img00352_jpg.rf.xxx.txt."""
    m = re.match(r"(MVI_\d+)", filename)
    return m.group(1) if m else ""


# ----- Main -----

def main():
    # Load report
    with open(REPORT) as f:
        report = json.load(f)

    flips = [i for i in report["issues"] if i["kind"] == "class_flip"]
    print(f"Loaded {len(flips)} class_flip issues from report")

    # Step 1: Build a list of "vehicle observations" from the flagged pairs.
    # Each flip gives us two observations of the same vehicle in two frames.
    # We group them into vehicle tracks using transitive IoU overlap.

    # Data structure: for each scene, collect all flagged vehicle observations
    # as (filename, bbox_tuple, class_id)
    scene_observations = defaultdict(list)  # scene -> [(file, bbox, class_id), ...]

    for flip in flips:
        scene = flip["scene"]
        detail = flip["detail"]

        # Parse classes from detail
        arrow_part = detail.split("(")[0].strip()
        class_a_name, class_b_name = [x.strip() for x in arrow_part.split("->")]
        class_a = NAME_TO_CLASS[class_a_name]
        class_b = NAME_TO_CLASS[class_b_name]

        # Parse boxes from detail
        boxes = re.findall(r"box_[ab]=\(([^)]+)\)", detail)
        bbox_a = parse_bbox(boxes[0])
        bbox_b = parse_bbox(boxes[1])

        scene_observations[scene].append((flip["file_a"], bbox_a, class_a))
        scene_observations[scene].append((flip["file_b"], bbox_b, class_b))

    # Step 2: For each scene, cluster observations into vehicle tracks.
    # Two observations are the same vehicle if their bboxes have IoU > threshold
    # and they're in different frames.
    # We use union-find for transitive closure.

    class UnionFind:
        def __init__(self, n):
            self.parent = list(range(n))
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        def union(self, a, b):
            a, b = self.find(a), self.find(b)
            if a != b:
                self.parent[b] = a

    # For each scene, build vehicle tracks
    scene_tracks = {}  # scene -> list of tracks, each track = [(file, bbox, class_id), ...]

    for scene, obs_list in scene_observations.items():
        # Deduplicate (same file+bbox might appear in multiple flips)
        unique_obs = {}
        for file, bbox, cls in obs_list:
            key = (file, bbox)
            if key not in unique_obs:
                unique_obs[key] = (file, bbox, cls)
        obs = list(unique_obs.values())

        uf = UnionFind(len(obs))
        for i in range(len(obs)):
            for j in range(i + 1, len(obs)):
                # Same vehicle if IoU > threshold AND different frames
                if obs[i][0] != obs[j][0] and iou(obs[i][1], obs[j][1]) > MATCH_IOU_THRESH:
                    uf.union(i, j)

        # Group by cluster
        clusters = defaultdict(list)
        for i, o in enumerate(obs):
            clusters[uf.find(i)].append(o)

        scene_tracks[scene] = list(clusters.values())

    total_tracks = sum(len(t) for t in scene_tracks.values())
    print(f"Identified {total_tracks} vehicle tracks across {len(scene_tracks)} scenes")

    # Step 3: For each vehicle track, scan ALL frames in the scene for matching bboxes.
    # This gives us majority vote across the full scene, not just the flagged pair.

    # Cache: scene -> list of all label files
    scene_files_cache = {}
    all_label_files = os.listdir(LABELS_DIR)
    for fname in all_label_files:
        s = extract_scene(fname)
        if s:
            if s not in scene_files_cache:
                scene_files_cache[s] = []
            scene_files_cache[s].append(fname)

    # For each track, compute the "reference position" as the average bbox center
    # Then scan all scene frames for matching bboxes
    changes = []  # list of (filepath, line_index, old_class, new_class)
    stats = Counter()  # (old_class, new_class) -> count

    for scene, tracks in scene_tracks.items():
        scene_file_list = scene_files_cache.get(scene, [])
        if not scene_file_list:
            continue

        # Pre-load all labels for this scene
        scene_labels = {}  # filename -> [(cls, cx, cy, w, h), ...]
        for fname in scene_file_list:
            scene_labels[fname] = read_label_file(LABELS_DIR / fname)

        for track in tracks:
            # Compute reference bbox as the average of all observations in the track
            avg_cx = sum(o[1][0] for o in track) / len(track)
            avg_cy = sum(o[1][1] for o in track) / len(track)
            avg_w = sum(o[1][2] for o in track) / len(track)
            avg_h = sum(o[1][3] for o in track) / len(track)
            ref_bbox = (avg_cx, avg_cy, avg_w, avg_h)

            # Scan all frames in this scene for bboxes matching this vehicle
            # Collect votes: (filename, line_index) -> class_id
            all_matches = []  # (filename, line_index, class_id, bbox)
            for fname, labels in scene_labels.items():
                for idx, lbl in enumerate(labels):
                    lbl_bbox = lbl[1:]
                    if iou(ref_bbox, lbl_bbox) > MATCH_IOU_THRESH:
                        all_matches.append((fname, idx, lbl[0], lbl_bbox))

            if not all_matches:
                continue

            # Majority vote
            class_votes = Counter(m[2] for m in all_matches)
            majority_class = class_votes.most_common(1)[0][0]
            total_votes = sum(class_votes.values())
            majority_count = class_votes[majority_class]

            # Only fix if there's a clear majority (> 50%)
            if majority_count <= total_votes / 2:
                continue

            # Fix minority frames — but ONLY for files that were flagged in the report.
            # We don't want to change labels that weren't flagged.
            flagged_files = set(o[0] for o in track)
            for fname, line_idx, cls, bbox in all_matches:
                if cls != majority_class and fname in flagged_files:
                    changes.append((fname, line_idx, cls, majority_class))
                    stats[(cls, majority_class)] += 1

    print(f"\nChanges to make: {len(changes)}")

    # Step 4: Apply changes — group by file, back up, edit
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    files_to_fix = defaultdict(list)  # filename -> [(line_idx, old_cls, new_cls), ...]
    for fname, line_idx, old_cls, new_cls in changes:
        files_to_fix[fname].append((line_idx, old_cls, new_cls))

    files_modified = 0
    labels_changed = 0

    for fname, fixes in files_to_fix.items():
        src = LABELS_DIR / fname
        bak = BACKUP_DIR / fname

        # Read original preserving raw text
        raw_labels = read_label_file_raw(src)
        if not raw_labels:
            continue

        # Backup (only if not already backed up)
        if not bak.exists():
            shutil.copy2(src, bak)

        # Apply fixes
        changed = False
        for line_idx, old_cls, new_cls in fixes:
            if line_idx < len(raw_labels) and raw_labels[line_idx][0] == old_cls:
                raw_labels[line_idx] = (new_cls, raw_labels[line_idx][1])
                changed = True
                labels_changed += 1

        if changed:
            write_label_file_raw(src, raw_labels)
            files_modified += 1

    # Report
    print(f"\n{'='*60}")
    print(f"CLASS FLIP FIX REPORT")
    print(f"{'='*60}")
    print(f"Labels changed:  {labels_changed}")
    print(f"Files modified:  {files_modified}")
    print(f"Backups saved:   {BACKUP_DIR}")
    print()
    print("Class changes breakdown:")
    for (old_c, new_c), count in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {CLASS_NAMES[old_c]:>5} -> {CLASS_NAMES[new_c]:<5}: {count}")
    print()

    # Verify: sanity check a few
    print("Verification (first 5 changed files):")
    for fname in list(files_to_fix.keys())[:5]:
        orig = read_label_file(BACKUP_DIR / fname)
        fixed = read_label_file(LABELS_DIR / fname)
        orig_classes = [CLASS_NAMES[l[0]] for l in orig]
        fixed_classes = [CLASS_NAMES[l[0]] for l in fixed]
        if orig_classes != fixed_classes:
            print(f"  {fname}")
            print(f"    before: {orig_classes}")
            print(f"    after:  {fixed_classes}")


if __name__ == "__main__":
    main()
