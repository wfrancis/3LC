use clap::Parser;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Local mAP@0.5 evaluator matching pycocotools COCO-style evaluation.
///
/// Reads YOLO-format prediction files (class conf x y w h) and ground-truth label files
/// (class x y w h), computes per-class AP at IoU=0.5 and the mean across classes.
#[derive(Parser)]
#[command(name = "map-eval", about = "Local mAP@0.5 evaluator (COCO-style)")]
struct Args {
    /// Directory containing prediction .txt files (YOLO format with confidence)
    #[arg(long)]
    predictions: PathBuf,

    /// Directory containing ground-truth label .txt files (YOLO format)
    #[arg(long)]
    labels: PathBuf,

    /// IoU threshold for matching
    #[arg(long, default_value = "0.5")]
    iou_threshold: f64,

    /// Number of classes
    #[arg(long, default_value = "4")]
    num_classes: u32,

    /// Image width (for absolute pixel conversion, 0 = stay normalized)
    #[arg(long, default_value = "0")]
    img_width: u32,

    /// Image height (for absolute pixel conversion, 0 = stay normalized)
    #[arg(long, default_value = "0")]
    img_height: u32,
}

#[derive(Debug, Clone)]
struct BBox {
    x_center: f64,
    y_center: f64,
    width: f64,
    height: f64,
}

#[derive(Debug, Clone)]
struct Detection {
    image: String,
    class_id: u32,
    confidence: f64,
    bbox: BBox,
}

#[derive(Debug, Clone)]
struct GroundTruth {
    image: String,
    class_id: u32,
    bbox: BBox,
    matched: bool,
}

#[derive(Debug, Serialize)]
struct ClassResult {
    class_id: u32,
    ap: f64,
    num_gt: usize,
    num_det: usize,
    num_tp: usize,
}

#[derive(Debug, Serialize)]
struct EvalResult {
    map_50: f64,
    per_class: Vec<ClassResult>,
    total_gt: usize,
    total_det: usize,
    num_images: usize,
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
    let union = a_area + b_area - inter_area;

    if union <= 0.0 { 0.0 } else { inter_area / union }
}

/// COCO-style 101-point interpolated AP
fn compute_ap(precisions: &[f64], recalls: &[f64]) -> f64 {
    if precisions.is_empty() {
        return 0.0;
    }

    // Add sentinel values
    let mut mrec = vec![0.0];
    mrec.extend_from_slice(recalls);
    mrec.push(1.0);

    let mut mpre = vec![0.0];
    mpre.extend_from_slice(precisions);
    mpre.push(0.0);

    // Make precision monotonically decreasing (right to left)
    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }

    // 101-point interpolation (COCO style)
    let mut ap = 0.0;
    for t in 0..=100 {
        let recall_thresh = t as f64 / 100.0;
        // Find precision at this recall threshold
        let mut p = 0.0;
        for j in 0..mrec.len() {
            if mrec[j] >= recall_thresh {
                p = mpre[j];
                break;
            }
        }
        ap += p;
    }
    ap / 101.0
}

fn compute_class_ap(
    detections: &mut [Detection],
    ground_truths: &mut [GroundTruth],
    iou_threshold: f64,
) -> (f64, usize) {
    let num_gt = ground_truths.len();
    if num_gt == 0 {
        return (0.0, 0);
    }

    // Sort detections by confidence descending
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

    // Reset matched flags
    for gt in ground_truths.iter_mut() {
        gt.matched = false;
    }

    // Group ground truths by image for fast lookup (store indices only)
    let mut gt_by_image: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, gt) in ground_truths.iter().enumerate() {
        gt_by_image
            .entry(gt.image.clone())
            .or_default()
            .push(i);
    }

    let mut tp_count = 0usize;
    let mut precisions = Vec::with_capacity(detections.len());
    let mut recalls = Vec::with_capacity(detections.len());

    for (det_idx, det) in detections.iter().enumerate() {
        if let Some(gt_indices) = gt_by_image.get(&det.image) {
            let gt_indices = gt_indices.clone();
            let mut best_iou = 0.0f64;
            let mut best_gt_idx: Option<usize> = None;

            for gi in &gt_indices {
                if ground_truths[*gi].matched {
                    continue;
                }
                let overlap = iou(&det.bbox, &ground_truths[*gi].bbox);
                if overlap > best_iou {
                    best_iou = overlap;
                    best_gt_idx = Some(*gi);
                }
            }

            if best_iou >= iou_threshold {
                if let Some(gi) = best_gt_idx {
                    ground_truths[gi].matched = true;
                    tp_count += 1;
                }
            }
        }

        let fp_count = (det_idx + 1) - tp_count;
        let precision = tp_count as f64 / (tp_count + fp_count) as f64;
        let recall = tp_count as f64 / num_gt as f64;
        precisions.push(precision);
        recalls.push(recall);
    }

    (compute_ap(&precisions, &recalls), tp_count)
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

    // Collect all detections and ground truths
    let mut all_detections: Vec<Detection> = Vec::new();
    let mut all_gts: Vec<GroundTruth> = Vec::new();

    let all_images: Vec<String> = {
        let mut imgs: Vec<String> = label_files.keys().cloned().collect();
        for k in pred_files.keys() {
            if !label_files.contains_key(k) {
                imgs.push(k.clone());
            }
        }
        imgs.sort();
        imgs
    };

    for image_name in &all_images {
        if let Some(content) = pred_files.get(image_name) {
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 6 {
                    if let (Ok(cls), Ok(conf), Ok(x), Ok(y), Ok(w), Ok(h)) = (
                        parts[0].parse::<u32>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                        parts[5].parse::<f64>(),
                    ) {
                        all_detections.push(Detection {
                            image: image_name.clone(),
                            class_id: cls,
                            confidence: conf,
                            bbox: BBox { x_center: x, y_center: y, width: w, height: h },
                        });
                    }
                }
            }
        }

        if let Some(content) = label_files.get(image_name) {
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    if let (Ok(cls), Ok(x), Ok(y), Ok(w), Ok(h)) = (
                        parts[0].parse::<u32>(),
                        parts[1].parse::<f64>(),
                        parts[2].parse::<f64>(),
                        parts[3].parse::<f64>(),
                        parts[4].parse::<f64>(),
                    ) {
                        all_gts.push(GroundTruth {
                            image: image_name.clone(),
                            class_id: cls,
                            bbox: BBox { x_center: x, y_center: y, width: w, height: h },
                            matched: false,
                        });
                    }
                }
            }
        }
    }

    // Compute per-class AP
    let mut per_class: Vec<ClassResult> = Vec::new();
    let mut sum_ap = 0.0;
    let mut num_classes_with_gt = 0;

    for cls in 0..args.num_classes {
        let mut cls_dets: Vec<Detection> = all_detections
            .iter()
            .filter(|d| d.class_id == cls)
            .cloned()
            .collect();
        let mut cls_gts: Vec<GroundTruth> = all_gts
            .iter()
            .filter(|g| g.class_id == cls)
            .cloned()
            .collect();

        let num_det = cls_dets.len();
        let num_gt = cls_gts.len();

        let (ap, tp) = compute_class_ap(&mut cls_dets, &mut cls_gts, args.iou_threshold);

        if num_gt > 0 {
            sum_ap += ap;
            num_classes_with_gt += 1;
        }

        per_class.push(ClassResult {
            class_id: cls,
            ap,
            num_gt,
            num_det,
            num_tp: tp,
        });
    }

    let map_50 = if num_classes_with_gt > 0 {
        sum_ap / num_classes_with_gt as f64
    } else {
        0.0
    };

    let result = EvalResult {
        map_50,
        per_class,
        total_gt: all_gts.len(),
        total_det: all_detections.len(),
        num_images: all_images.len(),
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());

    // Human-readable summary to stderr
    eprintln!("\n=== mAP@0.5 Evaluation ===");
    eprintln!("Images: {}", result.num_images);
    eprintln!("Ground truths: {}", result.total_gt);
    eprintln!("Detections: {}", result.total_det);
    eprintln!();
    let class_names = ["truck", "car", "van", "bus"];
    for c in &result.per_class {
        let name = class_names.get(c.class_id as usize).unwrap_or(&"?");
        eprintln!(
            "  {:<6} ({}): AP={:.4}  GT={:<5} Det={:<5} TP={}",
            name, c.class_id, c.ap, c.num_gt, c.num_det, c.num_tp
        );
    }
    eprintln!();
    eprintln!("  mAP@0.5 = {:.4}", result.map_50);
    eprintln!();
}
