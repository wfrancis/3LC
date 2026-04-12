use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// Propagate label fixes across neighboring video frames.
///
/// Given a fix manifest (JSON) describing corrections made on specific frames,
/// scans nearby frames in the same scene for labels matching the original error
/// (same class, IoU > threshold) and applies the same fix.
#[derive(Parser)]
#[command(name = "label-propagator", about = "Propagate label fixes across video frames")]
struct Args {
    /// Directory containing YOLO .txt label files
    #[arg(long)]
    labels: PathBuf,

    /// Path to the fix manifest JSON file
    #[arg(long)]
    fixes: PathBuf,

    /// Number of frames to scan in each direction from the fixed frame
    #[arg(long, default_value = "20")]
    range: u32,

    /// IoU threshold for matching a label to the fix location
    #[arg(long, default_value = "0.5")]
    iou_threshold: f64,

    /// Dry-run mode: report changes without applying them
    #[arg(long)]
    dry_run: bool,

    /// Output report path (default: stdout)
    #[arg(long)]
    output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Fix manifest types (input JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct FixManifest {
    fixes: Vec<Fix>,
}

#[derive(Debug, Clone, Deserialize)]
struct Fix {
    /// Scene identifier, e.g. "MVI_20011"
    scene: String,

    /// Frame number where the fix was originally made
    frame: u32,

    /// The original (incorrect) label
    old_label: FixLabel,

    /// The corrected label, or null/absent to indicate deletion
    new_label: Option<FixLabel>,
}

#[derive(Debug, Clone, Deserialize)]
struct FixLabel {
    /// Class id (0=truck, 1=car, 2=van, 3=bus)
    class_id: u32,

    /// Approximate YOLO-normalized coordinates
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

// ---------------------------------------------------------------------------
// Internal label representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BBox {
    class_id: u32,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

impl BBox {
    fn to_yolo_line(&self) -> String {
        format!(
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id, self.x_center, self.y_center, self.width, self.height
        )
    }
}

// ---------------------------------------------------------------------------
// Report types (output JSON)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct PropagationReport {
    total_fixes_in_manifest: usize,
    total_propagated: usize,
    total_frames_scanned: usize,
    fixes: Vec<FixReport>,
}

#[derive(Debug, Clone, Serialize)]
struct FixReport {
    scene: String,
    source_frame: u32,
    fix_type: String,
    propagated_to: Vec<FrameChange>,
}

#[derive(Debug, Clone, Serialize)]
struct FrameChange {
    frame: u32,
    file: String,
    label_line: usize,
    old_label: String,
    new_label: String,
    iou_with_fix: f64,
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

fn iou(a: &BBox, b: &BBox) -> f64 {
    let a_x1 = a.x_center - a.width / 2.0;
    let a_y1 = a.y_center - a.height / 2.0;
    let a_x2 = a.x_center + a.width / 2.0;
    let a_y2 = a.y_center + a.height / 2.0;

    let b_x1 = b.x_center - b.width / 2.0;
    let b_y1 = b.y_center - b.height / 2.0;
    let b_x2 = b.x_center + b.width / 2.0;
    let b_y2 = b.y_center + b.height / 2.0;

    let inter_x1 = a_x1.max(b_x1);
    let inter_y1 = a_y1.max(b_y1);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let a_area = a.width * a.height;
    let b_area = b.width * b.height;
    let union_area = a_area + b_area - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

fn fix_label_to_bbox(fl: &FixLabel) -> BBox {
    BBox {
        class_id: fl.class_id,
        x_center: fl.x_center,
        y_center: fl.y_center,
        width: fl.width,
        height: fl.height,
    }
}

// ---------------------------------------------------------------------------
// Label file I/O
// ---------------------------------------------------------------------------

fn parse_label_line(line: &str) -> Option<BBox> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return None;
    }
    Some(BBox {
        class_id: parts[0].parse().ok()?,
        x_center: parts[1].parse().ok()?,
        y_center: parts[2].parse().ok()?,
        width: parts[3].parse().ok()?,
        height: parts[4].parse().ok()?,
    })
}

fn read_labels(path: &Path) -> Vec<BBox> {
    match fs::read_to_string(path) {
        Ok(content) => content.lines().filter_map(parse_label_line).collect(),
        Err(_) => Vec::new(),
    }
}

fn write_labels(path: &Path, labels: &[BBox]) -> std::io::Result<()> {
    let content: String = labels
        .iter()
        .map(|b| b.to_yolo_line())
        .collect::<Vec<_>>()
        .join("\n");
    // Add trailing newline if there are labels
    let content = if content.is_empty() {
        content
    } else {
        format!("{}\n", content)
    };
    fs::write(path, content)
}

/// Build the filename for a given scene + frame number.
/// Convention: MVI_XXXXX__imgNNNNN.txt
fn frame_filename(scene: &str, frame: u32) -> String {
    format!("{scene}__img{frame:05}.txt")
}

// ---------------------------------------------------------------------------
// Backup
// ---------------------------------------------------------------------------

fn backup_file(path: &Path) -> std::io::Result<()> {
    let backup = path.with_extension("txt.bak");
    if !backup.exists() {
        // Only create backup if one doesn't already exist (don't overwrite
        // earlier backups with already-modified files)
        fs::copy(path, &backup)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scene index: discover which frames exist for each scene
// ---------------------------------------------------------------------------

/// Scan the labels directory and build a map: scene -> sorted list of frame numbers.
fn build_scene_index(labels_dir: &Path) -> HashMap<String, Vec<u32>> {
    let mut index: HashMap<String, Vec<u32>> = HashMap::new();

    let entries = match fs::read_dir(labels_dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("ERROR: cannot read labels directory {}: {}", labels_dir.display(), err);
            return index;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(true, |e| e != "txt") {
            continue;
        }
        let stem = match path.file_stem() {
            Some(s) => s.to_string_lossy().to_string(),
            None => continue,
        };

        // Expected format: MVI_XXXXX__imgNNNNN
        if let Some((scene, rest)) = stem.split_once("__img") {
            if let Ok(frame_num) = rest.parse::<u32>() {
                index.entry(scene.to_string()).or_default().push(frame_num);
            }
        }
    }

    // Sort frame numbers within each scene
    for frames in index.values_mut() {
        frames.sort_unstable();
    }

    index
}

// ---------------------------------------------------------------------------
// Core propagation logic
// ---------------------------------------------------------------------------

fn propagate_fix(
    fix: &Fix,
    labels_dir: &Path,
    scene_index: &HashMap<String, Vec<u32>>,
    iou_threshold: f64,
    range: u32,
    dry_run: bool,
) -> FixReport {
    let fix_type = if fix.new_label.is_some() {
        "class_change"
    } else {
        "delete"
    };

    let old_bbox = fix_label_to_bbox(&fix.old_label);

    let mut propagated_to: Vec<FrameChange> = Vec::new();

    // Determine frame range to scan
    let frames = match scene_index.get(&fix.scene) {
        Some(f) => f,
        None => {
            eprintln!(
                "WARNING: scene '{}' not found in labels directory",
                fix.scene
            );
            return FixReport {
                scene: fix.scene.clone(),
                source_frame: fix.frame,
                fix_type: fix_type.to_string(),
                propagated_to,
            };
        }
    };

    let min_frame = fix.frame.saturating_sub(range);
    let max_frame = fix.frame.saturating_add(range);

    for &frame_num in frames {
        // Skip the source frame itself (that was already fixed by hand)
        if frame_num == fix.frame {
            continue;
        }
        if frame_num < min_frame || frame_num > max_frame {
            continue;
        }

        let filename = frame_filename(&fix.scene, frame_num);
        let file_path = labels_dir.join(&filename);

        if !file_path.exists() {
            continue;
        }

        let labels = read_labels(&file_path);

        // Find label matching the old (erroneous) label: same class + high IoU
        let mut match_idx: Option<usize> = None;
        let mut best_iou: f64 = 0.0;

        for (i, label) in labels.iter().enumerate() {
            if label.class_id != old_bbox.class_id {
                continue;
            }
            let overlap = iou(label, &old_bbox);
            if overlap > best_iou {
                best_iou = overlap;
                match_idx = Some(i);
            }
        }

        // Did we find a matching label above the threshold?
        let idx = match match_idx {
            Some(i) if best_iou >= iou_threshold => i,
            _ => continue,
        };

        let old_line = labels[idx].to_yolo_line();

        // Build the new label set
        let (new_labels, new_line): (Vec<BBox>, String) = match &fix.new_label {
            None => {
                // Deletion: remove the matched label
                let mut new = labels.clone();
                new.remove(idx);
                (new, "DELETED".to_string())
            }
            Some(new_fix_label) => {
                // Class change / coordinate change: replace the matched label
                // Keep the ORIGINAL coordinates from this frame's label but apply
                // the class change. If the fix also moved the box, interpolate
                // proportionally (use the new label's coords directly for now;
                // a smarter version could interpolate based on frame distance).
                let replacement = BBox {
                    class_id: new_fix_label.class_id,
                    x_center: labels[idx].x_center,
                    y_center: labels[idx].y_center,
                    width: labels[idx].width,
                    height: labels[idx].height,
                };
                let new_line = replacement.to_yolo_line();
                let mut new = labels.clone();
                new[idx] = replacement;
                (new, new_line)
            }
        };

        propagated_to.push(FrameChange {
            frame: frame_num,
            file: filename.clone(),
            label_line: idx,
            old_label: old_line,
            new_label: new_line,
            iou_with_fix: best_iou,
        });

        // Apply the change (unless dry-run)
        if !dry_run {
            if let Err(e) = backup_file(&file_path) {
                eprintln!("WARNING: failed to backup {}: {}", file_path.display(), e);
            }
            if let Err(e) = write_labels(&file_path, &new_labels) {
                eprintln!("ERROR: failed to write {}: {}", file_path.display(), e);
            }
        }
    }

    // Sort by frame number for clean output
    propagated_to.sort_by_key(|c| c.frame);

    FixReport {
        scene: fix.scene.clone(),
        source_frame: fix.frame,
        fix_type: fix_type.to_string(),
        propagated_to,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = Args::parse();

    // Validate inputs
    if !args.labels.is_dir() {
        eprintln!("ERROR: labels directory '{}' does not exist", args.labels.display());
        std::process::exit(1);
    }
    if !args.fixes.is_file() {
        eprintln!("ERROR: fixes manifest '{}' does not exist", args.fixes.display());
        std::process::exit(1);
    }

    // Load fix manifest
    let manifest_text = match fs::read_to_string(&args.fixes) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read {}: {}", args.fixes.display(), e);
            std::process::exit(1);
        }
    };
    let manifest: FixManifest = match serde_json::from_str(&manifest_text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: invalid fix manifest JSON: {}", e);
            std::process::exit(1);
        }
    };

    eprintln!(
        "Loaded {} fixes from {}",
        manifest.fixes.len(),
        args.fixes.display()
    );

    // Build scene index
    let scene_index = build_scene_index(&args.labels);
    let total_scenes = scene_index.len();
    let total_frames: usize = scene_index.values().map(|v| v.len()).sum();
    eprintln!(
        "Indexed {} scenes, {} total frames in {}",
        total_scenes,
        total_frames,
        args.labels.display()
    );

    if args.dry_run {
        eprintln!("DRY RUN: no files will be modified");
    }

    // Process each fix
    let mut fix_reports: Vec<FixReport> = Vec::new();
    let mut total_propagated: usize = 0;
    let mut total_frames_scanned: usize = 0;

    for fix in &manifest.fixes {
        let report = propagate_fix(
            fix,
            &args.labels,
            &scene_index,
            args.iou_threshold,
            args.range,
            args.dry_run,
        );

        total_propagated += report.propagated_to.len();

        // Count frames scanned for this fix
        if let Some(frames) = scene_index.get(&fix.scene) {
            let min_frame = fix.frame.saturating_sub(args.range);
            let max_frame = fix.frame.saturating_add(args.range);
            total_frames_scanned += frames
                .iter()
                .filter(|&&f| f != fix.frame && f >= min_frame && f <= max_frame)
                .count();
        }

        fix_reports.push(report);
    }

    let report = PropagationReport {
        total_fixes_in_manifest: manifest.fixes.len(),
        total_propagated,
        total_frames_scanned,
        fixes: fix_reports,
    };

    // Output report
    let json = serde_json::to_string_pretty(&report).unwrap();
    match &args.output {
        Some(path) => {
            if let Err(e) = fs::write(path, &json) {
                eprintln!("ERROR: cannot write report to {}: {}", path.display(), e);
                std::process::exit(1);
            }
            eprintln!("Report written to {}", path.display());
        }
        None => {
            println!("{json}");
        }
    }

    // Summary to stderr
    eprintln!(
        "Done. {} fixes propagated across {} frames (from {} manifest entries).",
        total_propagated,
        total_frames_scanned,
        manifest.fixes.len()
    );
    if args.dry_run {
        eprintln!("(dry run -- no files were modified)");
    } else {
        eprintln!("Backups saved as .txt.bak alongside modified files.");
    }
}
