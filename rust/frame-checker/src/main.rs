use clap::Parser;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "frame-checker", about = "Check label consistency across sequential video frames")]
struct Cli {
    /// Path to the YOLO labels directory
    #[arg(long)]
    labels: PathBuf,

    /// Maximum number of issues to include in the report (ranked by severity)
    #[arg(long, default_value_t = 100)]
    top: usize,

    /// IoU threshold for matching boxes across frames
    #[arg(long, default_value_t = 0.3)]
    iou_threshold: f64,

    /// Size-jump ratio threshold (flag if size changes by more than this factor)
    #[arg(long, default_value_t = 0.5)]
    size_threshold: f64,

    /// Max frame-number gap to consider frames "consecutive" for analysis.
    /// Subsampled datasets may have large gaps between extracted frames;
    /// pairs separated by more than this many frame numbers are skipped.
    #[arg(long, default_value_t = 30)]
    max_frame_gap: u32,

    /// Class names (comma-separated, indexed by class_id)
    #[arg(long, default_value = "truck,car,van,bus")]
    class_names: String,
}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BBox {
    class_id: u32,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone)]
struct FrameLabels {
    scene: String,
    frame_num: u32,
    filename: String,
    boxes: Vec<BBox>,
}

#[derive(Debug, Clone, Serialize)]
struct Issue {
    severity: u32, // 1 = highest
    kind: String,
    scene: String,
    frame_a: u32,
    frame_b: Option<u32>,
    file_a: String,
    file_b: Option<String>,
    detail: String,
}

#[derive(Debug, Serialize)]
struct Report {
    total_files: usize,
    total_scenes: usize,
    total_boxes: usize,
    total_issues: usize,
    issues_by_kind: HashMap<String, usize>,
    issues: Vec<Issue>,
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn parse_filename(name: &str) -> Option<(String, u32)> {
    let re = Regex::new(r"^(MVI_\d+)_img(\d+)_jpg\.rf\.[a-f0-9]+\.txt$").unwrap();
    let caps = re.captures(name)?;
    let scene = caps.get(1)?.as_str().to_string();
    let frame: u32 = caps.get(2)?.as_str().parse().ok()?;
    Some((scene, frame))
}

fn parse_label_file(path: &Path) -> Vec<BBox> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
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
        })
        .collect()
}

fn load_all_frames(labels_dir: &Path) -> Vec<FrameLabels> {
    let entries: Vec<_> = fs::read_dir(labels_dir)
        .expect("Cannot read labels directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "txt")
                .unwrap_or(false)
        })
        .collect();

    entries
        .par_iter()
        .filter_map(|entry| {
            let filename = entry.file_name().to_string_lossy().to_string();
            let (scene, frame_num) = parse_filename(&filename)?;
            let boxes = parse_label_file(&entry.path());
            Some(FrameLabels {
                scene,
                frame_num,
                filename,
                boxes,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// IoU
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

/// Match boxes between two frames using class-agnostic IoU (greedy).
/// Returns pairs (idx_a, idx_b, iou_score) where IoU > threshold.
/// Class-agnostic so we can detect class flips.
fn match_boxes(frame_a: &[BBox], frame_b: &[BBox], threshold: f64) -> Vec<(usize, usize, f64)> {
    let mut scores: Vec<(usize, usize, f64)> = Vec::new();
    for (i, ba) in frame_a.iter().enumerate() {
        for (j, bb) in frame_b.iter().enumerate() {
            let s = iou(ba, bb);
            if s > threshold {
                scores.push((i, j, s));
            }
        }
    }
    // Greedy matching: highest IoU first, no double-assignment
    scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    let mut used_a = vec![false; frame_a.len()];
    let mut used_b = vec![false; frame_b.len()];
    let mut result = Vec::new();
    for (i, j, s) in scores {
        if !used_a[i] && !used_b[j] {
            used_a[i] = true;
            used_b[j] = true;
            result.push((i, j, s));
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

fn box_area(b: &BBox) -> f64 {
    b.width * b.height
}

fn class_name(id: u32, names: &[String]) -> String {
    names
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| format!("class_{}", id))
}

fn check_scene(
    frames: &[FrameLabels],
    iou_threshold: f64,
    size_threshold: f64,
    max_frame_gap: u32,
    class_names: &[String],
) -> Vec<Issue> {
    let mut issues = Vec::new();
    if frames.len() < 2 {
        return issues;
    }

    let scene = &frames[0].scene;

    // -----------------------------------------------------------------------
    // Build "close pairs": indices of consecutive frames within max_frame_gap
    // -----------------------------------------------------------------------
    let mut close_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..frames.len() - 1 {
        let gap = frames[i + 1].frame_num.saturating_sub(frames[i].frame_num);
        if gap <= max_frame_gap {
            close_pairs.push((i, i + 1));
        }
    }

    // -----------------------------------------------------------------------
    // Pass 1: Pairwise checks on close pairs (class flip + size jump)
    // -----------------------------------------------------------------------
    // forward_matches[(fi, fj)] = match list
    let mut forward_matches: HashMap<(usize, usize), Vec<(usize, usize, f64)>> = HashMap::new();

    for &(fi, fj) in &close_pairs {
        let fa = &frames[fi];
        let fb = &frames[fj];
        let matches = match_boxes(&fa.boxes, &fb.boxes, iou_threshold);

        for &(ai, bi, iou_score) in &matches {
            let ba = &fa.boxes[ai];
            let bb = &fb.boxes[bi];

            // (a) Class flip
            if ba.class_id != bb.class_id {
                issues.push(Issue {
                    severity: 1,
                    kind: "class_flip".to_string(),
                    scene: scene.clone(),
                    frame_a: fa.frame_num,
                    frame_b: Some(fb.frame_num),
                    file_a: fa.filename.clone(),
                    file_b: Some(fb.filename.clone()),
                    detail: format!(
                        "{} -> {} (IoU={:.3}, frame_gap={}) box_a=({:.4},{:.4},{:.4},{:.4}) box_b=({:.4},{:.4},{:.4},{:.4})",
                        class_name(ba.class_id, class_names),
                        class_name(bb.class_id, class_names),
                        iou_score,
                        fb.frame_num - fa.frame_num,
                        ba.x_center, ba.y_center, ba.width, ba.height,
                        bb.x_center, bb.y_center, bb.width, bb.height,
                    ),
                });
            }

            // (d) Size jump
            let area_a = box_area(ba);
            let area_b = box_area(bb);
            if area_a > 0.0 && area_b > 0.0 {
                let ratio = if area_a > area_b {
                    area_a / area_b
                } else {
                    area_b / area_a
                };
                // ratio > 1.5 means area changed by >50%
                if ratio > 1.0 / (1.0 - size_threshold) {
                    issues.push(Issue {
                        severity: 3,
                        kind: "size_jump".to_string(),
                        scene: scene.clone(),
                        frame_a: fa.frame_num,
                        frame_b: Some(fb.frame_num),
                        file_a: fa.filename.clone(),
                        file_b: Some(fb.filename.clone()),
                        detail: format!(
                            "{}: area {:.6} -> {:.6} (ratio={:.2}x, frame_gap={}) box_a=({:.4},{:.4},{:.4},{:.4}) box_b=({:.4},{:.4},{:.4},{:.4})",
                            class_name(ba.class_id, class_names),
                            area_a,
                            area_b,
                            ratio,
                            fb.frame_num - fa.frame_num,
                            ba.x_center, ba.y_center, ba.width, ba.height,
                            bb.x_center, bb.y_center, bb.width, bb.height,
                        ),
                    });
                }
            }
        }

        forward_matches.insert((fi, fj), matches);
    }

    // -----------------------------------------------------------------------
    // Pass 2: Build tracklets via chaining forward matches over close pairs
    // -----------------------------------------------------------------------
    let mut tracklets: Vec<Vec<(usize, usize)>> = Vec::new();
    let mut assigned: HashMap<(usize, usize), usize> = HashMap::new();

    for &(fi, fj) in &close_pairs {
        if let Some(matches) = forward_matches.get(&(fi, fj)) {
            for &(ai, bi, _) in matches {
                if let Some(&tid) = assigned.get(&(fi, ai)) {
                    tracklets[tid].push((fj, bi));
                    assigned.insert((fj, bi), tid);
                } else {
                    let tid = tracklets.len();
                    tracklets.push(vec![(fi, ai), (fj, bi)]);
                    assigned.insert((fi, ai), tid);
                    assigned.insert((fj, bi), tid);
                }
            }
        }
    }

    // Create singleton tracklets for unmatched boxes (only in frames that
    // participate in at least one close pair)
    let mut close_frame_set = std::collections::HashSet::new();
    for &(fi, fj) in &close_pairs {
        close_frame_set.insert(fi);
        close_frame_set.insert(fj);
    }

    for &fi in &close_frame_set {
        for bi in 0..frames[fi].boxes.len() {
            if !assigned.contains_key(&(fi, bi)) {
                let tid = tracklets.len();
                tracklets.push(vec![(fi, bi)]);
                assigned.insert((fi, bi), tid);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pass 3: Phantom single-frame labels and gap detection
    // -----------------------------------------------------------------------

    // Build neighbor map: for each frame index, which frame indices are close
    let mut has_close_prev: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut has_close_next: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for &(fi, fj) in &close_pairs {
        has_close_next.insert(fi);
        has_close_prev.insert(fj);
    }

    for tracklet in &tracklets {
        if tracklet.len() == 1 {
            let (fi, bi) = tracklet[0];
            // Only flag phantom if this frame has close neighbors on BOTH sides
            if has_close_prev.contains(&fi) && has_close_next.contains(&fi) {
                let b = &frames[fi].boxes[bi];
                issues.push(Issue {
                    severity: 2,
                    kind: "phantom_single_frame".to_string(),
                    scene: scene.clone(),
                    frame_a: frames[fi].frame_num,
                    frame_b: None,
                    file_a: frames[fi].filename.clone(),
                    file_b: None,
                    detail: format!(
                        "{}: appears only in this frame, not matched in neighbor frames (prev frame={}, next frame={}). box=({:.4},{:.4},{:.4},{:.4})",
                        class_name(b.class_id, class_names),
                        // find closest prev/next
                        frames.iter().enumerate()
                            .filter(|&(j, _)| j < fi && has_close_next.contains(&j))
                            .map(|(_, f)| f.frame_num)
                            .max()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "?".to_string()),
                        frames.iter().enumerate()
                            .filter(|&(j, _)| j > fi && has_close_prev.contains(&j))
                            .map(|(_, f)| f.frame_num)
                            .min()
                            .map(|n| n.to_string())
                            .unwrap_or_else(|| "?".to_string()),
                        b.x_center, b.y_center, b.width, b.height,
                    ),
                });
            }
        }

        // Gap detection: tracklet skips a frame that exists in the close-pair chain
        if tracklet.len() >= 2 {
            for w in tracklet.windows(2) {
                let (fi_a, bi_a) = w[0];
                let (fi_b, _bi_b) = w[1];
                // Check if there is an intermediate frame index between fi_a and fi_b
                // that is within close-pair distance of both
                for mid in (fi_a + 1)..fi_b {
                    // mid must be close to fi_a (fi_a->mid is a close pair)
                    // and close to fi_b (mid->fi_b is a close pair)
                    let close_to_a = close_pairs.iter().any(|&(a, b)| a == fi_a && b == mid);
                    let close_to_b = close_pairs.iter().any(|&(a, b)| a == mid && b == fi_b);
                    if close_to_a && close_to_b {
                        let ba = &frames[fi_a].boxes[bi_a];
                        issues.push(Issue {
                            severity: 2,
                            kind: "gap_detection".to_string(),
                            scene: scene.clone(),
                            frame_a: frames[fi_a].frame_num,
                            frame_b: Some(frames[fi_b].frame_num),
                            file_a: frames[fi_a].filename.clone(),
                            file_b: Some(frames[fi_b].filename.clone()),
                            detail: format!(
                                "{}: present in frame {} and {}, missing in frame {} ({})",
                                class_name(ba.class_id, class_names),
                                frames[fi_a].frame_num,
                                frames[fi_b].frame_num,
                                frames[mid].frame_num,
                                frames[mid].filename,
                            ),
                        });
                    }
                }
            }
        }
    }

    issues
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();
    let class_names: Vec<String> = cli
        .class_names
        .split(',')
        .map(|s| s.trim().to_string())
        .collect();

    eprintln!("Loading label files from {:?} ...", cli.labels);
    let all_frames = load_all_frames(&cli.labels);
    let total_boxes: usize = all_frames.iter().map(|f| f.boxes.len()).sum();
    eprintln!(
        "Loaded {} files, {} total boxes",
        all_frames.len(),
        total_boxes,
    );

    // Group by scene
    let mut scenes: HashMap<String, Vec<FrameLabels>> = HashMap::new();
    for frame in all_frames {
        scenes.entry(frame.scene.clone()).or_default().push(frame);
    }

    // Sort each scene by frame number, then deduplicate by frame_num
    // (augmented copies have same frame_num but different hash; keep first)
    for frames in scenes.values_mut() {
        frames.sort_by_key(|f| f.frame_num);
        frames.dedup_by_key(|f| f.frame_num);
    }

    let total_scenes = scenes.len();
    let deduped_files: usize = scenes.values().map(|v| v.len()).sum();
    eprintln!(
        "Found {} scenes, {} unique frames (after dedup)",
        total_scenes, deduped_files
    );

    // Analyze each scene in parallel
    let mut all_issues: Vec<Issue> = scenes
        .par_iter()
        .flat_map(|(_, frames)| {
            check_scene(
                frames,
                cli.iou_threshold,
                cli.size_threshold,
                cli.max_frame_gap,
                &class_names,
            )
        })
        .collect();

    // Sort by severity (ascending = most severe first), then by scene+frame
    all_issues.sort_by(|a, b| {
        a.severity
            .cmp(&b.severity)
            .then_with(|| a.scene.cmp(&b.scene))
            .then_with(|| a.frame_a.cmp(&b.frame_a))
    });

    let total_issues = all_issues.len();

    // Count by kind
    let mut issues_by_kind: HashMap<String, usize> = HashMap::new();
    for issue in &all_issues {
        *issues_by_kind.entry(issue.kind.clone()).or_insert(0) += 1;
    }

    eprintln!("Found {} total issues:", total_issues);
    for (kind, count) in &issues_by_kind {
        eprintln!("  {}: {}", kind, count);
    }

    // Truncate to --top
    all_issues.truncate(cli.top);

    let report = Report {
        total_files: deduped_files,
        total_scenes,
        total_boxes,
        total_issues,
        issues_by_kind,
        issues: all_issues,
    };

    let json = serde_json::to_string_pretty(&report).expect("JSON serialization failed");
    println!("{}", json);
}
