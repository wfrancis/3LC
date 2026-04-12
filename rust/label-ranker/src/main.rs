use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Compare model predictions against YOLO labels to rank images by label error likelihood.
///
/// Reads YOLO-format prediction files (class conf x y w h per line) and label files
/// (class x y w h per line), computes error signals, and outputs a ranked JSON report.
#[derive(Parser)]
#[command(name = "label-ranker", about = "Rank images by label error likelihood")]
struct Args {
    /// Directory containing prediction .txt files (YOLO format with confidence)
    #[arg(long)]
    predictions: PathBuf,

    /// Directory containing ground-truth label .txt files (YOLO format)
    #[arg(long)]
    labels: PathBuf,

    /// IoU threshold below which a prediction is considered unmatched
    #[arg(long, default_value = "0.3")]
    iou_unmatched: f64,

    /// IoU threshold below which a matched prediction is considered inaccurate
    #[arg(long, default_value = "0.5")]
    iou_inaccurate: f64,

    /// Minimum confidence for predictions to be considered
    #[arg(long, default_value = "0.25")]
    min_conf: f64,

    /// IoU threshold above which two same-class labels are considered duplicates
    #[arg(long, default_value = "0.7")]
    duplicate_iou: f64,

    /// Minimum box area (fraction of image) below which a label is flagged as tiny
    #[arg(long, default_value = "0.001")]
    min_area: f64,

    /// Number of top results to output (0 = all)
    #[arg(long, short, default_value = "100")]
    top: usize,

    /// Output format: json or tsv
    #[arg(long, default_value = "json")]
    format: String,
}

#[derive(Debug, Clone)]
struct BBox {
    class_id: u32,
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone)]
struct Prediction {
    bbox: BBox,
    confidence: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Issue {
    issue_type: String,
    detail: String,
    severity: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ImageReport {
    image: String,
    score: f64,
    num_labels: usize,
    num_predictions: usize,
    issues: Vec<Issue>,
}

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

fn parse_prediction_line(line: &str) -> Option<Prediction> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 6 {
        return None;
    }
    Some(Prediction {
        bbox: BBox {
            class_id: parts[0].parse().ok()?,
            x_center: parts[2].parse().ok()?,
            y_center: parts[3].parse().ok()?,
            width: parts[4].parse().ok()?,
            height: parts[5].parse().ok()?,
        },
        confidence: parts[1].parse().ok()?,
    })
}

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

fn analyze_image(
    image_name: &str,
    labels: &[BBox],
    predictions: &[Prediction],
    args: &Args,
) -> ImageReport {
    let mut issues: Vec<Issue> = Vec::new();

    // Track which labels have been matched
    let mut label_matched = vec![false; labels.len()];

    // 1. For each high-confidence prediction, find best matching label
    let filtered_preds: Vec<&Prediction> = predictions
        .iter()
        .filter(|p| p.confidence >= args.min_conf)
        .collect();

    for pred in &filtered_preds {
        let mut best_iou = 0.0f64;
        let mut best_idx: Option<usize> = None;
        let mut best_class_match = false;

        for (i, label) in labels.iter().enumerate() {
            let overlap = iou(&pred.bbox, label);
            if overlap > best_iou {
                best_iou = overlap;
                best_idx = Some(i);
                best_class_match = pred.bbox.class_id == label.class_id;
            }
        }

        if best_iou < args.iou_unmatched {
            // High-confidence prediction with no matching label = MISSING ANNOTATION
            issues.push(Issue {
                issue_type: "missing_label".into(),
                detail: format!(
                    "class={} conf={:.2} at ({:.3},{:.3}) {}x{} — no matching label (best IoU={:.2})",
                    pred.bbox.class_id, pred.confidence,
                    pred.bbox.x_center, pred.bbox.y_center,
                    pred.bbox.width, pred.bbox.height, best_iou
                ),
                severity: pred.confidence * (1.0 - best_iou),
            });
        } else if best_iou < args.iou_inaccurate {
            // Matched but low IoU = INACCURATE BOX
            if let Some(idx) = best_idx {
                label_matched[idx] = true;
            }
            issues.push(Issue {
                issue_type: "inaccurate_box".into(),
                detail: format!(
                    "class={} conf={:.2} IoU={:.2} — label box is imprecise",
                    pred.bbox.class_id, pred.confidence, best_iou
                ),
                severity: pred.confidence * (1.0 - best_iou) * 0.8,
            });
        } else {
            // Good match
            if let Some(idx) = best_idx {
                label_matched[idx] = true;
                // Check class mismatch
                if !best_class_match {
                    issues.push(Issue {
                        issue_type: "wrong_class".into(),
                        detail: format!(
                            "pred_class={} label_class={} conf={:.2} IoU={:.2} — class mismatch at same location",
                            pred.bbox.class_id, labels[idx].class_id, pred.confidence, best_iou
                        ),
                        severity: pred.confidence * best_iou,
                    });
                }
            }
        }
    }

    // 2. Check for duplicate labels (same class, high overlap)
    for i in 0..labels.len() {
        for j in (i + 1)..labels.len() {
            if labels[i].class_id == labels[j].class_id {
                let overlap = iou(&labels[i], &labels[j]);
                if overlap > args.duplicate_iou {
                    issues.push(Issue {
                        issue_type: "duplicate_label".into(),
                        detail: format!(
                            "class={} labels {} and {} overlap IoU={:.2}",
                            labels[i].class_id, i, j, overlap
                        ),
                        severity: overlap,
                    });
                }
            }
        }
    }

    // 3. Check for tiny/garbage boxes
    for (i, label) in labels.iter().enumerate() {
        let area = label.width * label.height;
        if area < args.min_area {
            issues.push(Issue {
                issue_type: "tiny_box".into(),
                detail: format!(
                    "class={} label {} area={:.6} — suspiciously small",
                    label.class_id, i, area
                ),
                severity: (args.min_area - area) / args.min_area,
            });
        }
    }

    // 4. Labels that no prediction matched at all (possible phantom labels)
    for (i, matched) in label_matched.iter().enumerate() {
        if !matched && !filtered_preds.is_empty() {
            issues.push(Issue {
                issue_type: "unmatched_label".into(),
                detail: format!(
                    "class={} label {} at ({:.3},{:.3}) — no prediction matched it",
                    labels[i].class_id, i, labels[i].x_center, labels[i].y_center
                ),
                severity: 0.5,
            });
        }
    }

    let score: f64 = issues.iter().map(|i| i.severity).sum();

    ImageReport {
        image: image_name.to_string(),
        score,
        num_labels: labels.len(),
        num_predictions: predictions.len(),
        issues,
    }
}

fn load_txt_files(dir: &Path) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "txt") {
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                if let Ok(content) = fs::read_to_string(&path) {
                    map.insert(stem, content);
                }
            }
        }
    }
    map
}

fn main() {
    let args = Args::parse();

    let pred_files = load_txt_files(&args.predictions);
    let label_files = load_txt_files(&args.labels);

    // Get all image names (union of predictions and labels)
    let mut all_images: Vec<String> = label_files.keys().cloned().collect();
    for k in pred_files.keys() {
        if !label_files.contains_key(k) {
            all_images.push(k.clone());
        }
    }
    all_images.sort();

    let mut reports: Vec<ImageReport> = all_images
        .par_iter()
        .map(|image_name| {
            let labels: Vec<BBox> = label_files
                .get(image_name)
                .map(|content| {
                    content
                        .lines()
                        .filter_map(parse_label_line)
                        .collect()
                })
                .unwrap_or_default();

            let predictions: Vec<Prediction> = pred_files
                .get(image_name)
                .map(|content| {
                    content
                        .lines()
                        .filter_map(parse_prediction_line)
                        .collect()
                })
                .unwrap_or_default();

            analyze_image(image_name, &labels, &predictions, &args)
        })
        .collect();

    // Sort by score descending (worst first)
    reports.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let top = if args.top == 0 {
        reports.len()
    } else {
        args.top.min(reports.len())
    };
    let reports = &reports[..top];

    match args.format.as_str() {
        "tsv" => {
            println!("rank\timage\tscore\tnum_labels\tnum_preds\tissues");
            for (i, r) in reports.iter().enumerate() {
                let issue_summary: Vec<String> = r.issues.iter().map(|i| i.issue_type.clone()).collect();
                println!(
                    "{}\t{}\t{:.3}\t{}\t{}\t{}",
                    i + 1,
                    r.image,
                    r.score,
                    r.num_labels,
                    r.num_predictions,
                    issue_summary.join(",")
                );
            }
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(reports).unwrap());
        }
    }

    eprintln!(
        "Analyzed {} images, reported top {} by error likelihood",
        all_images.len(),
        top
    );
}
